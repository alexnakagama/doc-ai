from pydantic import BaseModel


class Prompt(BaseModel):
    system: str
    user: str
