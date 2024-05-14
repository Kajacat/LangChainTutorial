from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings()
video_url = "https://www.youtube.com/watch?v=_ZvnD73m40o&ab_channel=freeCodeCamp.org"


def create_vector_db_from_youtube(url):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    hub_llm = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        model_kwargs={"temperature": 0.9, "max_length": 100},
    )

    prompt = PromptTemplate(input_variables=["docs"], template="{docs}")

    chain = LLMChain(llm=hub_llm, prompt=prompt)
    response = chain.run(docs=docs_page_content)
    response = response.replace("\n", " ")
    return response
