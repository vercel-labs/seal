
class Storage:
    # abstract storage layer, can be backed by
    # postgres, json, in-memory
    # used by session tree as backend
    async def write():
        pass
    async def read():
        pass
