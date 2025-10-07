import logging 
import re
import os
import unicodedata
import math
import asyncio
import inspect
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import translators as ts
    HAS_TRANSLATORS = True
except Exception:
    ts = None
    HAS_TRANSLATORS = False

from markitdown import MarkItDown
from zh_normalization import text_normlization
from config import EBOOK_SETTINGS

PUNCT_CHINESE = "，。！？；：、（）【】《》「」『』……——·"
PUNCT_LATIN = ",.?!;:/-—–_"
SENTENCE_END = "。！？.!?"
_RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
_RE_INLINE_CODE = re.compile(r'```.*?```', flags=re.S)
_RE_MARKDOWN_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_RE_IMAGE_MD = re.compile(r'!\[[^\]]+\]\([^)]+\)')
_RE_ENGLISH_PATTERN = re.compile(r'\b[a-zA-Z]+(?:[\'-]?[a-zA-Z]+)*(?:\s+[a-zA-Z]+(?:[\'-]?[a-zA-Z]+)*)*\b')
_RE_CHINESE_CHARS = re.compile(r'[\u4e00-\u9fff]')

_PUNC_PIPE_CACHE = {"pipe": None, "model_dir": None}

@lru_cache(maxsize=4)
def _import_modelscope_pipeline() -> Optional[Tuple[Callable, object]]:
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        return pipeline, Tasks
    except Exception as e:
        logger.debug("[AI] modelscope import failed: %s", e)
        return None

@lru_cache(maxsize=1)
def get_punc_pipe(model_dir: Path):
    model_dir_s = str(model_dir)
    if _PUNC_PIPE_CACHE["pipe"] and _PUNC_PIPE_CACHE["model_dir"] == model_dir_s:
        return _PUNC_PIPE_CACHE["pipe"]
    imported = _import_modelscope_pipeline()
    if not imported:
        raise RuntimeError("modelscope not available")
    pipeline, Tasks = imported
    try:
        p = pipeline(task=Tasks.punctuation, model=model_dir_s, model_revision="v2.0.4")
        _PUNC_PIPE_CACHE["pipe"] = p
        _PUNC_PIPE_CACHE["model_dir"] = model_dir_s
        logger.info("[AI] Punctuation model loaded from %s", model_dir_s)
        return p
    except Exception as e:
        logger.error("[AI] Failed to instantiate punctuation pipeline from %s: %s", model_dir_s, e)
        raise

def _clean_punctuation_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'([，。！？：；、])\1+', r'\1', text)
    text = re.sub(r'([，！？：；、])。', r'\1', text)
    text = re.sub(r'。([，！？：；、])', r'\1', text)
    text = re.sub(r'([！？，；：])\s*([,!?:;])', r'\1', text)
    text = re.sub(r'([,!?:;])\s*([！？，；：])', r'\2', text)
    while True:
        new_text = re.sub(r'([，。！？：；、,!?:;])\s*([，。！？：；、,!?:;])', r'\2', text)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r'[，。！？：；、,!?:;]{2,}', lambda m: m.group()[-1], text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = _RE_MULTIPLE_NEWLINES.sub('\n\n', text)
    return text

def ai_add_punctuation_with_original(text: str, model_id: Optional[str] = None, model_dir: Optional[Path] = None) -> Tuple[str, str]:
    if not text or not text.strip():
        return text, text
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    punc_pipe = get_punc_pipe(model_dir)
    lines = text.splitlines()
    outputs: List[str] = []
    for line in lines:
        if not line.strip():
            outputs.append(line)
            continue
        try:
            rec = punc_pipe(input=line)
            if inspect.isawaitable(rec):
                raise RuntimeError("modelscope pipeline returned awaitable")
            if isinstance(rec, list):
                punctuated = "".join([r.get("text", "") if isinstance(r, dict) else str(r) for r in rec])
            elif isinstance(rec, dict):
                punctuated = rec.get("text", line)
            else:
                punctuated = str(rec)
        except Exception as e:
            logger.warning("[AI] Punctuation failed for '%s...': %s", line[:30], e)
            outputs.append(line)
            continue
        outputs.append(punctuated)
    result = "\n".join(outputs)
    return text, result

def _align_punctuation_keep_original(original: str, predicted: str) -> str:
    if not original or not predicted:
        return predicted or original or ""
    PUNCTS = set("，。！？；：、（）【】《》,.?!;:()[]\"'""''")
    orig_chars = []
    orig_punct_after = {}
    char_idx = 0
    for i, c in enumerate(original):
        if c not in PUNCTS and not c.isspace():
            orig_chars.append(c)
            if i + 1 < len(original) and original[i + 1] in PUNCTS:
                orig_punct_after[char_idx] = original[i + 1]
            char_idx += 1
    pred_chars = []
    pred_punct_after = {}
    char_idx = 0
    for i, c in enumerate(predicted):
        if c not in PUNCTS and not c.isspace():
            pred_chars.append(c)
            if i + 1 < len(predicted) and predicted[i + 1] in PUNCTS:
                pred_punct_after[char_idx] = predicted[i + 1]
            char_idx += 1
    result = []
    for i, char in enumerate(pred_chars):
        result.append(char)
        if i in orig_punct_after:
            result.append(orig_punct_after[i])
        elif i in pred_punct_after:
            result.append(pred_punct_after[i])
    return ''.join(result)

def _clean_predicted_double_punctuation(text: str) -> str:
    if not text:
        return text
    HIGH_PRIORITY = {'。', '！', '？', '.', '!', '?'}
    LOW_PRIORITY = {'，', '；', '：', '、', ',', ';', ':'}
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in HIGH_PRIORITY or char in LOW_PRIORITY:
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char in HIGH_PRIORITY or next_char in LOW_PRIORITY:
                    if char in HIGH_PRIORITY:
                        result.append(char)
                    elif next_char in HIGH_PRIORITY:
                        result.append(next_char)
                        i += 1
                    else:
                        result.append(char)
                    i += 1
                    continue
        result.append(char)
        i += 1
    text = ''.join(result)
    text = re.sub(r'([，。！？：；、])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def ai_add_punctuation(text: str, model_id: Optional[str] = None, model_dir: Optional[Path] = None) -> str:
    if not text or not text.strip():
        return text
    original_text, predicted_text = ai_add_punctuation_with_original(text, model_id, model_dir)
    aligned_text = _align_punctuation_keep_original(original_text, predicted_text)
    final_text = _clean_predicted_double_punctuation(aligned_text)
    final_text = _clean_punctuation_text(final_text)
    final_text = _RE_MULTIPLE_NEWLINES.sub('\n\n', final_text)
    return final_text

class EbookProcessor:
    def __init__(self, enable_punctuation_alignment: bool = True, debug_alignment: bool = False):
        self.chapter_patterns = EBOOK_SETTINGS.get("chapter_detection_patterns", [])
        self.model_token_limits: Dict[str, int] = {
            "xtts": 380,
            "kokoro": 100,
            "chattts": 170,
            "cosyvoice": 380
        }
        self.max_tokens: int = 380

        self.markitdown = MarkItDown()
        self.text_normalizer = text_normlization.TextNormalizer()
        self.tokenizer = None
        self.avg_chars_per_token = 1.5
        self.enable_punctuation_alignment = enable_punctuation_alignment
        self.debug_alignment = debug_alignment
        self.translator = ts if HAS_TRANSLATORS else None
        self.punc_model_dir = Path(__file__).parent.parent / "models" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

    def token_count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            try:
                enc = None
                try:
                    enc = self.tokenizer.encode(text, add_special_tokens=False)
                except Exception:
                    enc = self.tokenizer(text)
                    if isinstance(enc, dict):
                        enc = enc.get("input_ids") or enc.get("ids") or enc.get("inputIds")
                if inspect.isawaitable(enc):
                    logger.warning("[TOKEN] tokenizer returned awaitable; falling back to char-estimate.")
                elif enc is not None:
                    try:
                        return len(enc)
                    except Exception:
                        logger.debug("[TOKEN] len(enc) failed, fallback to chars.")
            except Exception as e:
                logger.debug("[TOKEN] tokenizer error: %s", e)
        return max(1, int(math.ceil(len(text) / self.avg_chars_per_token)))

    def load_tokenizer(self, tokenizer_dir: str, model_id: Optional[str] = None):
        try:
            from transformers import AutoTokenizer, AutoConfig
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            logger.info("[OK] tokenizer loaded from %s", tokenizer_dir)
            
            # 尝试从 config 获取 max token
            try:
                config = AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)
                max_len = getattr(config, "max_position_embeddings", None) \
                          or getattr(config, "max_sequence_length", None) \
                          or getattr(self.tokenizer, "model_max_length", None)
                if isinstance(max_len, int) and max_len > 0:
                    if model_id:
                        self.model_token_limits[model_id] = max_len
                        logger.info("[TOKEN] model %s max tokens set to %d (from config)", model_id, max_len)
            except Exception as e:
                logger.debug("[TOKEN] no max length info in config: %s", e)

        except Exception as e:
            logger.warning("[WARN] cannot load tokenizer from %s: %s", tokenizer_dir, e)
            self.tokenizer = None

    def set_model(self, model_id: str):
        self.max_tokens = self.model_token_limits.get(model_id, self.max_tokens)
        logger.info("[MODEL] using %s with max_tokens=%d", model_id, self.max_tokens)




    async def process_ebook(self, ebook_path: str) -> List[Dict]:
        ebook_path = Path(ebook_path)
        if not ebook_path.exists():
            raise FileNotFoundError(f"Ebook file not found: {ebook_path}")
        markdown_text = await self._convert_to_markdown(ebook_path)
        if inspect.isawaitable(markdown_text):
            logger.error("[BUG] _convert_to_markdown returned awaitable")
            raise RuntimeError("_convert_to_markdown returned coroutine; expected string.")
        chapters = self._split_into_chapters(markdown_text)
        return chapters

    async def _convert_to_markdown(self, file_path: Path) -> str:
        try:
            md = MarkItDown()
            conv = getattr(md, "convert", None)
            if inspect.iscoroutinefunction(conv):
                result = await conv(str(file_path))
            else:
                result = await asyncio.to_thread(conv, str(file_path))
            markdown_text = getattr(result, "text_content", str(result))
            
            if HAS_TRANSLATORS and self.translator:
                try:
                    markdown_text = await self._translate_english_to_chinese(markdown_text)
                except Exception as e:
                    logger.warning("[TRANSLATE] translation failed: %s", e)
            
            if self.enable_punctuation_alignment:
                try:
                    markdown_text = self._apply_punctuation_alignment(markdown_text)
                except Exception as e:
                    logger.debug("[PUNC] ai_add_punctuation failed: %s", e)
            
            normalized_text = await asyncio.to_thread(self._normalize_text_sync, markdown_text)
            
            return normalized_text
        except Exception as e:
            logger.error("[ERROR] Failed to convert %s: %s", file_path, e)
            raise

    def _normalize_text_sync(self, text: str) -> str:
        try:
            if self._contains_english_words(text):
                return text
            if not self._looks_chinese(text):
                return text
            normalized_text = self.text_normalizer.normalize_sentence(text)
            if not normalized_text.strip() or not self._has_content_chars(normalized_text):
                return text
            return normalized_text
        except Exception:
            return text

    def _has_content_chars(self, s: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s))

    def _looks_chinese(self, s: str) -> bool:
        han = re.findall(r"[\u4e00-\u9fff]", s)
        return len(han) >= 10 or (han and len(han) / max(1, len(s)) > 0.1)

    def _contains_english_words(self, s: str) -> bool:
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', s)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', s)
        if english_words and (not chinese_chars or len(english_words) / max(1, len(chinese_chars)) > 0.05):
            return True
        return False

    async def _translate_english_to_chinese(self, text: str) -> str:
        if not self.translator or not text or not text.strip():
            return text
        try:
            matches = list(_RE_ENGLISH_PATTERN.finditer(text))
            if not matches:
                return text
            english_phrases = []
            valid_matches = []
            for m in matches:
                eng = m.group(0)
                if eng.isupper() and len(eng) <= 5:
                    continue
                if eng.isdigit():
                    continue
                english_phrases.append(eng)
                valid_matches.append(m)
            if not english_phrases:
                return text
            translations = await asyncio.to_thread(self._batch_translate_phrases_sync, english_phrases)
            result_text = text
            for m, translation in zip(reversed(valid_matches), reversed(translations)):
                if translation and translation != m.group(0):
                    try:
                        start, end = m.start(), m.end()
                        if 0 <= start < end <= len(result_text):
                            result_text = result_text[:start] + translation + result_text[end:]
                    except Exception as e:
                        logger.debug("[TRANSLATE] replacement failed for '%s': %s", m.group(0), e)
                        continue
            return result_text
        except Exception as e:
            logger.warning("[TRANSLATE] translation step failed: %s", e)
            return text

    def _batch_translate_phrases_sync(self, phrases: List[str]) -> List[str]:
        translations = []
        for phrase in phrases:
            translated = None
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    if attempt == 0:
                        translated = self.translator.translate_text(phrase, translator='bing', from_language='en', to_language='zh')
                    else:
                        translated = self.translator.translate_text(phrase, translator='google', from_language='en', to_language='zh')
                    if translated and str(translated).strip() != phrase:
                        break
                except Exception as e:
                    logger.debug("[TRANSLATE] attempt %d failed for '%s': %s", attempt + 1, phrase, e)
                    if attempt == max_retries - 1:
                        translated = phrase
            translations.append(str(translated) if translated else phrase)
        return translations

    def _apply_punctuation_alignment(self, text: str) -> str:
        if not self.enable_punctuation_alignment:
            return text
        
        if not isinstance(text, str):
            raise RuntimeError("_apply_punctuation_alignment expected str")
        
        try:
            if len(text) > 5000:
                parts_original = []
                parts_predicted = []
                for i in range(0, len(text), 3000):
                    chunk = text[i:i+3000]
                    orig, pred = ai_add_punctuation_with_original(chunk, model_dir=self.punc_model_dir)
                    parts_original.append(orig)
                    parts_predicted.append(pred)
                original_text = "".join(parts_original)
                predicted_text = "".join(parts_predicted)
            else:
                original_text, predicted_text = ai_add_punctuation_with_original(text, model_dir=self.punc_model_dir)
            
            aligned_text = _align_punctuation_keep_original(original_text, predicted_text)
            final_text = _clean_predicted_double_punctuation(aligned_text)
            final_text = _clean_punctuation_text(final_text)
            return final_text
            
        except Exception:
            return text

    def _align_with_original(self, original: str, predicted: str) -> str:
        """
        将AI预测的标点与原文对齐，优先保留原文中的标点符号
        """
        if not original or not predicted:
            return predicted or original or ""
        
        result = predicted
        result = re.sub(r'！\s*，', '！', result)
        result = re.sub(r'？\s*，', '？', result)
        result = re.sub(r'。\s*，', '。', result)
        result = re.sub(r'，\s*！', '！', result)
        result = re.sub(r'，\s*？', '？', result)
        result = re.sub(r'，\s*。', '。', result)
        
        return result

    def _split_into_chapters(self, markdown_text: str) -> List[Dict]:
        if not isinstance(markdown_text, str):
            logger.error("[BUG] markdown_text is not a string")
            raise RuntimeError("markdown_text must be str")
        if not markdown_text.strip():
            return [{"title": "Empty Content", "text": "", "order": 1}]
        lines = markdown_text.splitlines()
        chapters: List[Dict] = []
        current_chapter: Dict[str, Any] = {"title": "Chapter 1", "text": "", "order": 1}
        current_content: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_content:
                    current_content.append("")
                continue
            chapter_title = self._detect_chapter_title(line)
            if chapter_title:
                if current_content:
                    text_content = "\n".join(current_content).strip()
                    text_content = self.sanitize_for_tts(text_content)
                    if text_content:
                        current_chapter["text"] = text_content
                        chapters.append(current_chapter)
                current_chapter = {"title": chapter_title, "text": "", "order": len(chapters) + 1}
                current_content = [chapter_title, ""]
            else:
                current_content.append(line)
        if current_content:
            text_content = "\n".join(current_content).strip()
            text_content = self.sanitize_for_tts(text_content)
            if text_content:
                current_chapter["text"] = text_content
                chapters.append(current_chapter)
        if not chapters:
            clean_text = self.sanitize_for_tts(markdown_text.strip())
            chapters = [{"title": "Full Book", "text": clean_text, "order": 1}]
        return chapters

    def _detect_chapter_title(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        if re.match(r"^#+\s+.+", line):
            return re.sub(r"^#+\s*", "", line).strip()
        if re.match(r"^\d+[\.)]\s*.+", line) or re.match(r"^第[一二三四五六七八九十百千万\d]+[章回节].+", line):
            return line.strip()
        for pattern in self.chapter_patterns:
            try:
                if re.match(pattern, line, re.IGNORECASE):
                    return re.sub(r"^#+\s*|\d+[\.)]\s*", "", line).strip()
            except re.error:
                continue
        return None

    def split_chapter(self, chapter_text: str, model_id: str = None) -> List[str]:
        if inspect.isawaitable(chapter_text):
            logger.error("[BUG] split_chapter received coroutine instead of str")
            raise RuntimeError("split_chapter received coroutine; ensure caller awaited any async function.")
        if not chapter_text or not chapter_text.strip():
            return []

        cleaned_text = self.sanitize_for_tts(chapter_text)
        if model_id:
            self.set_model(model_id)
        safe_limit = self.max_tokens

        conservative_chars_per_token_map = {
            "chattts": 1.0,
        }
        conservative_chars = conservative_chars_per_token_map.get(model_id, min(self.avg_chars_per_token, 1.2))
        conservative_chars = max(0.8, min(conservative_chars, self.avg_chars_per_token))

        sentences: List[str] = []
        paragraphs = self._split_paragraphs(cleaned_text)
        for para in paragraphs:
            if not para.strip():
                continue
            parts = re.split(r'(?<=[。！？.!?])\s*', para)
            parts = [p.strip() for p in parts if p and p.strip()]
            if parts:
                sentences.extend(parts)
            else:
                sentences.append(para.strip())

        final_fragments: List[str] = []
        pause_indicators = ['但是','然而','虽然','尽管','因此','所以','由于','因为','此外','而且','同时','接着','然后','最后','首先','其次','例如','例如：']

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue

            if self.token_count(s) <= safe_limit:
                if len(s) / conservative_chars <= safe_limit * 1.0:
                    final_fragments.append(s)
                    continue

            comma_parts = [p.strip() for p in re.split(r'(?<=[,，;；、])\s*', s) if p.strip()]
            if len(comma_parts) > 1:
                for cp in comma_parts:
                    if self.token_count(cp) <= safe_limit and len(cp) / conservative_chars <= safe_limit:
                        final_fragments.append(cp)
                    else:
                        final_fragments.extend(self._chunk_by_token_estimate(cp, safe_limit, chars_per_token=conservative_chars))
                continue

            found_pause = False
            for word in pause_indicators:
                pos = s.find(word, max(8, int(len(s)*0.12)), max(20, int(len(s)*0.6)))
                if pos != -1:
                    left = s[:pos].strip()
                    right = s[pos:].strip()
                    if left:
                        if self.token_count(left) <= safe_limit and len(left) / conservative_chars <= safe_limit:
                            final_fragments.append(left)
                        else:
                            final_fragments.extend(self._chunk_by_token_estimate(left, safe_limit, chars_per_token=conservative_chars))
                    if right:
                        if self.token_count(right) <= safe_limit and len(right) / conservative_chars <= safe_limit:
                            final_fragments.append(right)
                        else:
                            final_fragments.extend(self._chunk_by_token_estimate(right, safe_limit, chars_per_token=conservative_chars))
                    found_pause = True
                    break
            if found_pause:
                continue

            if ' ' in s:
                pieces = [p.strip() for p in re.split(r'\s+', s) if p.strip()]
                acc = ""
                for piece in pieces:
                    cand = (acc + " " + piece).strip() if acc else piece
                    if len(cand) / conservative_chars <= safe_limit:
                        acc = cand
                    else:
                        if acc:
                            final_fragments.append(acc)
                        acc = piece
                if acc:
                    final_fragments.append(acc)
                continue

            final_fragments.extend(self._chunk_by_token_estimate(s, safe_limit, chars_per_token=conservative_chars))

        merged = self._merge_short_fragments(final_fragments, min_len=30, max_chars=int(safe_limit * conservative_chars * 1.1))

        final: List[str] = []
        for frag in merged:
            if self.token_count(frag) <= safe_limit and len(frag) / conservative_chars <= safe_limit:
                final.append(frag.strip())
            else:
                final.extend(self._chunk_by_token_estimate(frag, safe_limit, chars_per_token=conservative_chars))

        return [f for f in (s.strip() for s in final) if f]

    def _chunk_by_token_estimate(self, text: str, token_limit: int, chars_per_token: Optional[float] = None) -> List[str]:
        if not text:
            return []
        if chars_per_token is None:
            chars_per_token = self.avg_chars_per_token
        est_chars = max(20, int(token_limit * chars_per_token))
        n = len(text)
        res: List[str] = []
        cur = 0
        while cur < n:
            remain = n - cur
            if remain <= est_chars:
                part = text[cur:].strip()
                if part:
                    res.append(part)
                break
            end = cur + est_chars
            window = text[cur:end]
            split_at = None
            for sep in ("。", "！", "？", "，", ",", ";", "；", "、", " "):
                pos = window.rfind(sep)
                if pos != -1 and pos >= int(est_chars * 0.25):
                    split_at = cur + pos + 1
                    break
            if split_at is None or split_at <= cur:
                split_at = end
            piece = text[cur:split_at].strip()
            if piece:
                res.append(piece)
            if split_at <= cur:
                cur = cur + est_chars
            else:
                cur = split_at

        final: List[str] = []
        for p in res:
            if self.token_count(p) <= token_limit and len(p) / chars_per_token <= token_limit:
                final.append(p)
            else:
                approx_len = int(math.ceil(len(p) / max(1, math.ceil(self.token_count(p) / token_limit))))
                i = 0
                while i < len(p):
                    part = p[i:i+approx_len].strip()
                    if part:
                        final.append(part)
                    i += approx_len
        return final

    def _merge_short_fragments(self, fragments: List[str], min_len: int = 30, max_chars: Optional[int] = None) -> List[str]:
        out: List[str] = []
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue
            if out and len(frag) < min_len:
                if max_chars is None or len(out[-1]) + 1 + len(frag) <= max_chars:
                    out[-1] = out[-1] + " " + frag
                else:
                    out.append(frag)
            else:
                out.append(frag)
        return out

    def sanitize_for_tts(self, s: str) -> str:
        """
        仅清理 markdown/inline code/image 等，不要破坏标点
        """
        if not s:
            return ""
        s = _RE_IMAGE_MD.sub("", s)
        s = _RE_MARKDOWN_LINK.sub(r'\1', s)
        s = _RE_INLINE_CODE.sub("", s)
        s = re.sub(r"^\s{0,3}([>\-+*])\s*", "", s, flags=re.M)
        s = re.sub(r"(\*\*|__|\*|_)", "", s)
        replacements = {"•": "", "·": "", "●": "", "→": "，", "＋": "加", "＝": "等于", "~": "到"}
        for k, v in replacements.items():
            s = s.replace(k, v)
        s = _RE_MULTIPLE_NEWLINES.sub("\n\n", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = _clean_punctuation_text(s)
        return s.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs: List[str] = []
        current_paragraph: List[str] = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                current_paragraph.append(line)
            elif current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        return paragraphs

    def _is_punctuation(self, ch: str) -> bool:
        if not ch:
            return False
        if ch.isspace():
            return False
        try:
            cat = unicodedata.category(ch)
            if cat and cat.startswith('P'):
                return True
        except Exception:
            pass
        if ch in (PUNCT_CHINESE + PUNCT_LATIN + "\"'“”‘’"):
            return True
        return False