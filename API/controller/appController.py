from API.service.appService import Service
from fastapi import HTTPException, Request, Response

class Controller:

    async def ask_bot(request: Request):
        body = await request.json()
        query = body.get("query")
        model_name = body.get("model_name")
        rag_type = body.get("rag_type")
        preprocessing_type = body.get("preprocessing_type")
        cohere_hybrid = body.get("cohere_hybrid")
        summary_flag = body.get("summary_flag")

        if not model_name or not query or not rag_type or not preprocessing_type:
            raise HTTPException(status_code=400, detail="Missing parameters")

        try:
            response = await Service.ask_bot(query, model_name, rag_type, preprocessing_type, cohere_hybrid, summary_flag)
            # return {"response": response}
            formatted_response = f"""{response['response']}

            {response['relevance score']}  {response['faithfulness score']}
            """
            return Response(content=formatted_response, media_type="text/markdown")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))