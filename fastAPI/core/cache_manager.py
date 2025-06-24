import hashlib

# Simple in-memory cache to avoid repeated API calls
_query_cache = {}
_cache_max_size = 50

# Audio cache for TTS
_audio_cache = {}
_audio_cache_max_size = 20 # TTS audio files can be larger

def get_query_hash(query_text):
    """Generate a hash for caching queries"""
    return hashlib.md5(query_text.encode()).hexdigest()

def cache_result(query_hash, result):
    """Cache query result"""
    global _query_cache
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available"""
    return _query_cache.get(query_hash)

def get_audio_hash(text, lang, speed=1.0):
    """Generate hash for audio caching including speed"""
    combined = f"{text}_{lang}_{speed}"
    return hashlib.md5(combined.encode()).hexdigest()

def cache_audio(audio_hash, audio_data):
    """Cache audio data"""
    global _audio_cache
    if len(_audio_cache) >= _audio_cache_max_size:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(_audio_cache))
        del _audio_cache[oldest_key]
    _audio_cache[audio_hash] = audio_data

def get_cached_audio(audio_hash):
    """Get cached audio if available"""
    return _audio_cache.get(audio_hash)

def get_audio_cache_stats():
    """Get audio cache statistics"""
    # This is a simplified version. A more accurate hit rate would require tracking hits vs. misses.
    # For now, returning size and max size.
    return {
        'total_audio_cached': len(_audio_cache),
        'audio_cache_max_size': _audio_cache_max_size,
        'total_queries_cached': len(_query_cache),
        'query_cache_max_size': _cache_max_size,
    }

def clear_audio_cache():
    global _audio_cache
    _audio_cache.clear()
    print("Audio cache cleared.")