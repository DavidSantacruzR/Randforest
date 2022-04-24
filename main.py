from fastapi import FastAPI
import uvicorn

classifier = FastAPI()


@classifier.get("/")
def welcome():
    return {"Hello": "There"}


if __name__ == "__main":
    uvicorn.run("main:classifier", host="127.0.0.1", port=3000, log_level="info")
