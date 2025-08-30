from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence
import pyopenjtalk


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin

def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


# TODO: Need to probaby specify the prefix character in config or something, 
# or do something more robust.
# TODO: Figure out a way to deal with symbols and numbers using the context to determine if they go through g2p or not.
def convert_char_to_phonemes(text_list, polyphone=True):
    """
    Convert a list of text strings into OpenJTalk phoneme sequences.
    Each phoneme is prefixed with the '¤' symbol. 
    
    Mixed text is split into segments, with only Japanese parts processed through g2p.
    English segments are skipped (return empty lists).

    Args:
        text_list (list[str]): Input list of sentences.
        polyphone (bool): kept for API compatibility (not used in OpenJTalk).
    
    Returns:
        list[list[str]]: List of phoneme sequences with each phoneme prefixed by '¤'.
        English segments return empty lists.
    
    Examples:
        >>> convert_char_to_phonemes(["こんにちは", "お元気ですか"])
        [["¤k", "¤o", "¤N", "¤n", "¤i", "¤ch", "¤i", "¤w", "¤a"], 
         ["¤o", "¤g", "¤e", "¤n", "¤k", "¤i", "¤d", "¤e", "¤s", "¤u", "¤k", "¤a"]]
        
        >>> convert_char_to_phonemes(["Today I took the 新幹線 to Tokyo"])
        [['T', 'o', 'd', 'a', 'y', ' ', 'I', ' ', 't', 'o', 'o', 'k', ' ', 't', 'h', 'e', ' ', 
        '¤sh', '¤i', '¤N', '¤k', '¤a', '¤N', '¤s', '¤e', '¤N', ' ', 't', 'o', ' ', 'T', 'o', 'k', 'y', 'o']]
    """
    final_text_list = []

    for text in text_list:
        result = []
        segments = split_japanese_segments(text)

        for segment in segments:
            if segment["is_japanese"]:
                phonemes = pyopenjtalk.g2p(segment["text"], kana=False).strip().split()
                result.extend(f"¤{p}" for p in phonemes)
            else:
                # Keep English/raw chars as-is
                result.extend(segment["text"])  

        final_text_list.append(result)

    return final_text_list


def split_japanese_segments(text):
    """
    Split text into segments with language identification.
    
    Args:
        text (str): Input text with potential mixed content
        
    Returns:
        list[dict]: List of segment dictionaries with text and language info
    
    Example:
        >>> split_japanese_segments("Today I took the 新幹線 to Tokyo! 100回ぐらい乗りました! Amazing!")
        [
            {"text": "Today I took the ", "is_japanese": False, "language": "en"},
            {"text": "新幹線", "is_japanese": True, "language": "ja"},
            {"text": " to Tokyo! ", "is_japanese": False, "language": "en"},
            {"text": "100回ぐらい乗りました!", "is_japanese": True, "language": "ja"},
            {"text": " Amazing!", "is_japanese": False, "language": "en"}
        ]
    """
    segments = []
    current_segment = ""
    current_is_japanese = None

    def is_symbol_or_number(c):
        code = ord(c)
        # digits 0-9
        if 48 <= code <= 57:
            return True
        # specific symbols: ! ? . , -
        if code in (33, 63, 46, 44, 45):
            return True
        return False

    # Precompute the "next real character" info
    text_len = len(text)
    next_real_is_japanese = [False] * text_len
    next_japanese = None
    for i in reversed(range(text_len)):
        c = text[i]
        if not is_symbol_or_number(c):
            next_japanese = is_japanese_char(c)
        next_real_is_japanese[i] = next_japanese if next_japanese is not None else False

    # Build segments
    for i, char in enumerate(text):
        if is_symbol_or_number(char):
            is_japanese = next_real_is_japanese[i]
        else:
            is_japanese = is_japanese_char(char)

        # detect boundary
        if current_is_japanese is not None and is_japanese != current_is_japanese:
            segments.append({
                "text": current_segment,
                "is_japanese": current_is_japanese,
                "language": "ja" if current_is_japanese else "en"
            })
            current_segment = ""

        current_segment += char
        current_is_japanese = is_japanese

    if current_segment:
        segments.append({
            "text": current_segment,
            "is_japanese": current_is_japanese,
            "language": "ja" if current_is_japanese else "en"
        })

    return segments

def is_japanese_char(char):
    """
    Check if a single character is a Japanese character by examining its Unicode code point.
    
    Args:
        char (str): A single character to check
        
    Returns:
        bool: True if the character is Japanese, False otherwise
    """
    if len(char) != 1:
        return False
    
    # Get the Unicode code point of the character
    code_point = ord(char)
    
    # Check against Japanese Unicode blocks:
    
    # 1. CJK Symbols and Punctuation (3000-303F)
    # Includes: Japanese-specific punctuation, brackets, repetition marks
    # Examples: 。、・「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩〪〭〮〯〫〬
    if 0x3000 <= code_point <= 0x303F:
        return True
    
    # 2. Hiragana (3040-309F)
    # Includes: All hiragana characters, small hiragana, combining marks
    # Examples: あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん
    #           ぁぃぅぇぉゃゅょっゎゐゑゔゕゖ゙゚゛゜ゝゞゟ
    if 0x3040 <= code_point <= 0x309F:
        return True
    
    # 3. Katakana (30A0-30FF)
    # Includes: All katakana characters, small katakana, half-width katakana, katakana punctuation
    # Examples: アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン
    #           ァィゥェォャュョッヮヰヱヵヶヷヸヹヺ・ーヽヾヿ
    if 0x30A0 <= code_point <= 0x30FF:
        return True
    
    # 4. CJK Unified Ideographs (4E00-9FFF) - Common Kanji
    # Includes: The main block of CJK unified ideographs (Kanji/Hanzi)
    # Note: This range includes Chinese characters too, but in Japanese context,
    #       they are used as Kanji. Context determines language.
    # Examples: 一丁七万丈三上下不且世丘丙両並中丸主久乏乗乙九乳乾乱了事二云互五井亜亡交享京人仁今介仕他付代令以仮仲件任休会
    if 0x4E00 <= code_point <= 0x9FFF:
        return True
    
    # 5. Halfwidth and Fullwidth Forms (FF00-FFEF)
    # Includes: Full-width ASCII variants, full-width katakana, full-width punctuation
    # Examples: 
    #   - Full-width ASCII: ！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～
    #   - Full-width katakana: ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ
    if 0xFF00 <= code_point <= 0xFFEF:
        return True
    
    # Additional Japanese-specific ranges (less common but worth noting):
    
    # 6. CJK Compatibility Ideographs (F900-FAFF) - Rare/obsolete kanji
    # if 0xF900 <= code_point <= 0xFAFF:
    #     return True
    
    # 7. CJK Unified Ideographs Extension A (3400-4DBF) - Less common kanji
    # if 0x3400 <= code_point <= 0x4DBF:
    #     return True
    
    # 8. CJK Unified Ideographs Extension B (20000-2A6DF) - Very rare kanji
    # if 0x20000 <= code_point <= 0x2A6DF:
    #     return True
    
    # 9. CJK Unified Ideographs Extension C-F (2A700-2B73F, 2B740-2B81F, 2B820-2CEAF, 2CEB0-2EBEF)
    # if 0x2A700 <= code_point <= 0x2EBEF:
    #     return True
    
    # 10. CJK Compatibility Ideographs Supplement (2F800-2FA1F)
    # if 0x2F800 <= code_point <= 0x2FA1F:
    #     return True
    
    return False


if __name__ == "__main__":
    #print(convert_char_to_phonemes(["こんにちは", "お元気ですか", "Today I took the 新幹線 to Tokyo"]))
    #print(convert_char_to_pinyin(["hello this is a test", "what will I get"]))
    print(convert_char_to_phonemes(["Today I took the 新幹線 to Tokyo! 100回ぐらい乗りました! Amazing!"]))