import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Test it
r.set('test', 'Hello Redis!')
print(r.get('test'))  # Should print: Hello Redis!