from typing import List, Optional, Awaitable
import asyncio

class StringFuture:
    def __init__(self, future: str | Awaitable[str], context: Optional[str] = None):
        self._future = future
        self._context = context
        self._val: Optional[str] = None
    
    def is_str(self) -> bool:
        return isinstance(self._future, str)
    
    def is_future(self) -> bool:
        return isinstance(self._future, Awaitable)
    
    async def wait(self) -> str:
        if self._val is not None:
            return self._val
        if self.is_str():
            self._val = self._future
        elif self.is_future():
            self._val = await self._future
        else:
            raise ValueError()
        return self._val
                