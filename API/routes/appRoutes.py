from API.controller.appController import Controller
from fastapi import APIRouter, Request

router = APIRouter(prefix="/complaince", tags=["ARCH"])

@router.post("/bot/ask")
async def ask_bot(request: Request):
    return await Controller.ask_bot(request)