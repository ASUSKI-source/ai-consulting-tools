import re
from pathlib import Path
import PyPDF2


def extract_text(file_path):
    """
    Extract text from various file formats (.txt, .pdf, .md).
    Returns cleaned text with normalized whitespace and paragraph breaks.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt' or suffix == '.md':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
    
    elif suffix == '.pdf':
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                if pdf_reader.is_encrypted:
                    raise ValueError(f'PDF file is encrypted and cannot be read: {file_path.name}')
                
                pages = []
                for page in pdf_reader.pages:
                    pages.append(page.extract_text())
                
                text = '\n\n'.join(pages)
        except Exception as e:
            if 'encrypted' in str(e).lower():
                raise ValueError(f'PDF file is encrypted and cannot be read: {file_path.name}')
            raise
    
    else:
        raise ValueError(f'Unsupported file type: {suffix}')
    
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def split_into_sentences(text):
    """
    Split text into sentences using regex pattern.
    Pattern matches whitespace after sentence-ending punctuation (.!?)
    The lookbehind (?<=...) ensures we split AFTER the punctuation, not before.
    """
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s for s in sentences if s.strip()]


def smart_chunk_text(text, target_words=400, min_words=80, overlap_sentences=2):
    """
    Chunk text intelligently at paragraph/sentence boundaries with overlap.
    
    Strategy:
    1. Split into paragraphs (double newline)
    2. Process each paragraph, respecting sentence boundaries
    3. Build chunks up to target_words
    4. When saving a chunk, carry over the last overlap_sentences to the next chunk
    5. Never cut mid-sentence
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk_sentences = []
    
    def word_count(text_list):
        return sum(len(s.split()) for s in text_list)
    
    def save_chunk():
        """Save current chunk and return sentences for overlap."""
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            # Return the last N sentences for overlap with next chunk
            # This ensures continuity between chunks and helps with context
            if len(current_chunk_sentences) > overlap_sentences:
                return current_chunk_sentences[-overlap_sentences:]
            else:
                return current_chunk_sentences.copy()
        return []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        para_words = len(paragraph.split())
        
        if para_words <= target_words:
            current_chunk_sentences.append(paragraph)
            
            if word_count(current_chunk_sentences) >= target_words:
                overlap = save_chunk()
                current_chunk_sentences = overlap
        else:
            sentences = split_into_sentences(paragraph)
            
            for sentence in sentences:
                current_chunk_sentences.append(sentence)
                
                if word_count(current_chunk_sentences) >= target_words:
                    overlap = save_chunk()
                    current_chunk_sentences = overlap
    
    if current_chunk_sentences and word_count(current_chunk_sentences) >= min_words:
        chunks.append(' '.join(current_chunk_sentences))
    
    return chunks


def chunk_file_smart(file_path, target_words=400, min_words=80, overlap_sentences=2):
    """
    Extract text from file and chunk it smartly with metadata.
    Returns list of dicts with chunk text and metadata.
    """
    text = extract_text(file_path)
    chunks = smart_chunk_text(text, target_words, min_words, overlap_sentences)
    
    file_path = Path(file_path)
    total_chunks = len(chunks)
    
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            'text': chunk_text,
            'metadata': {
                'source_file': file_path.name,
                'chunk_index': i,
                'total_chunks': total_chunks,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'starts_with': chunk_text[:50],
            }
        })
    
    return result


def chunk_text_word_count(text, chunk_size=400, overlap=50):
    """
    Old approach: simple word-count chunking without respecting boundaries.
    This is the Week 5 logic for comparison purposes.
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(' '.join(chunk_words))
        i += chunk_size - overlap
    
    return chunks


def compare_chunkers(text):
    """
    Compare old word-count approach vs new smart approach.
    Shows metrics including mid-sentence cuts and boundary quality.
    """
    old_chunks = chunk_text_word_count(text, chunk_size=400, overlap=50)
    new_chunks = smart_chunk_text(text, target_words=400, min_words=80, overlap_sentences=2)
    
    def count_mid_sentence_cuts(chunks):
        """Count chunks that don't end with sentence-ending punctuation."""
        count = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and chunk[-1] not in '.!?':
                count += 1
        return count
    
    def avg_words(chunks):
        if not chunks:
            return 0
        return sum(len(c.split()) for c in chunks) / len(chunks)
    
    old_cuts = count_mid_sentence_cuts(old_chunks)
    new_cuts = count_mid_sentence_cuts(new_chunks)
    
    def boundary_quality(cuts, total):
        if total == 0:
            return 'N/A'
        ratio = cuts / total
        if ratio == 0:
            return 'Good'
        elif ratio < 0.3:
            return 'OK'
        else:
            return 'Poor'
    
    print("\n" + "="*80)
    print("CHUNKING COMPARISON")
    print("="*80)
    print(f"{'Method':<15} | {'Chunks':<6} | {'Avg Words':<10} | {'Mid-sent Cuts':<15} | {'Quality':<10}")
    print("-"*80)
    print(f"{'Word-count':<15} | {len(old_chunks):<6} | {avg_words(old_chunks):<10.1f} | {old_cuts:<15} | {boundary_quality(old_cuts, len(old_chunks)):<10}")
    print(f"{'Smart':<15} | {len(new_chunks):<6} | {avg_words(new_chunks):<10.1f} | {new_cuts:<15} | {boundary_quality(new_cuts, len(new_chunks)):<10}")
    print("="*80 + "\n")


if __name__ == '__main__':
    sample_text = """
    Artificial intelligence has transformed the way we interact with technology. Machine learning algorithms 
    can now process vast amounts of data in seconds. This capability has opened up new possibilities in 
    fields ranging from healthcare to finance.
    
    In healthcare, AI systems can analyze medical images with remarkable accuracy. They can detect patterns 
    that human doctors might miss. These systems don't replace doctors, but rather augment their capabilities. 
    The result is faster, more accurate diagnoses that can save lives.
    
    The financial sector has also embraced AI technology. Trading algorithms can execute thousands of 
    transactions per second. Risk assessment models can predict market trends with increasing precision. 
    Banks use AI to detect fraudulent transactions in real-time. This protects both the institutions and 
    their customers from financial losses.
    
    Natural language processing has made significant strides in recent years. Chatbots can now handle 
    customer service inquiries with human-like responses. Translation services can convert text between 
    languages almost instantaneously. Voice assistants understand context and can carry on meaningful 
    conversations with users.
    
    However, these advances come with important ethical considerations. Privacy concerns arise when AI 
    systems collect and analyze personal data. Bias in training data can lead to unfair outcomes. There 
    are questions about accountability when AI systems make decisions that affect people's lives. Society 
    must grapple with these issues as the technology continues to evolve.
    
    The future of AI holds both promise and uncertainty. Researchers are working on more sophisticated 
    models that can understand and reason about the world. There's potential for AI to help solve global 
    challenges like climate change and disease. But we must ensure that this powerful technology is 
    developed and deployed responsibly, with human values at the center of the conversation.
    """
    
    print("\n" + "="*80)
    print("SMART CHUNKER DEMO")
    print("="*80)
    
    compare_chunkers(sample_text)
    
    print("\n" + "="*80)
    print("SMART CHUNKER - BOUNDARY VISUALIZATION")
    print("="*80)
    
    chunks = smart_chunk_text(sample_text, target_words=150, min_words=50, overlap_sentences=1)
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ({len(chunk.split())} words) ---")
        chunk_display = chunk[:200] + ('...' if len(chunk) > 200 else '')
        print(chunk_display)
        
        last_char = chunk.strip()[-1] if chunk.strip() else ''
        boundary = "✓ CLEAN" if last_char in '.!?' else "✗ CUT MID-SENTENCE"
        print(f"Ends with: '{last_char}' [{boundary}]")
    
    print("\n" + "="*80)
