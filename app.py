from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# === 1. Setup Environment & Inisialisasi Model ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# === Global Dictionaries untuk Menyimpan Memory & Pipeline per User ===
user_booking_memory = {} 
user_global_memory = {}  
user_active_pipeline = {} 

# === 2. Intent Classification Chain (Global) ===
intent_prompt_template = """
User message: {user_query}

Klasifikasikan pesan tersebut ke dalam salah satu kategori berikut:
- greetings: jika user menyapa.
- goodbye: jika user mengucapkan selamat tinggal.
- information: jika user meminta informasi umum tentang dokter atau jadwal.
- booking: jika user ingin melakukan booking appointment dokter.
- out of topic: jika pesan tidak terkait dengan layanan dokter.

Jawab hanya dengan salah satu kategori di atas.
"""
intent_prompt = PromptTemplate(
    input_variables=["user_query"],
    template=intent_prompt_template
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# === 3. Dummy Data Dokter ===
doctors = [
    {"name": "Dr. John Smith", "specialist": "Cardiology", "schedule": "Monday 9-12, Wednesday 14-18"},
    {"name": "Dr. Emily Rose", "specialist": "Dermatology", "schedule": "Tuesday 10-14, Thursday 9-12"},
    {"name": "Dr. Alex Green", "specialist": "Pediatrics", "schedule": "Friday 10-16"}
]
doctor_data_str = "\n".join([
    f"Name: {doc['name']}, Specialist: {doc['specialist']}, Schedule: {doc['schedule']}"
    for doc in doctors
])

# === 4. Chain untuk Informasi Dokter (tanpa memory) ===
doctor_prompt_template = """
Anda adalah chatbot yang membantu manajemen dokter. Berikut adalah daftar dokter beserta detailnya:

{doctor_data}

User: {user_query}
Chatbot:"""
doctor_prompt = PromptTemplate(
    input_variables=["doctor_data", "user_query"],
    template=doctor_prompt_template
)
doctor_chain = LLMChain(llm=llm, prompt=doctor_prompt)

# === 5. Booking Appointment Chain (dengan Conversation Memory) ===
booking_prompt_template = f"""
Percakapan sebelumnya:
{{chat_history}}

Anda adalah asisten booking appointment dokter. Ikuti langkah-langkah berikut untuk memproses booking:
1. Evaluasi detail permintaan booking berdasarkan percakapan sebelumnya dan pesan user.
2. Periksa apakah waktu yang diminta tersedia berdasarkan jadwal dokter.
3. Jika waktu tidak tersedia, sarankan waktu alternatif yang tersedia.
4. Jika detail booking sudah lengkap (nama dokter, tanggal, dan waktu), tanyakan konfirmasi booking dengan menanyakan: "Apakah Anda ingin mengonfirmasi booking dengan detail tersebut?"
5. Buat respons yang ringkas dan jelas.

Berikut daftar dokter:
{doctor_data_str}

User: {{user_query}}

Chatbot:"""
booking_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=booking_prompt_template
)

# === 6. Membuat API dengan FastAPI ===
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

def process_user_query(user_id: str, message: str) -> str:
    # Jika pesan "/reset", reset memory untuk user tersebut
    if message.strip() == "/reset":
        user_booking_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
        user_global_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
        user_active_pipeline[user_id] = None
        return "Memory telah direset."

    # Pastikan setiap user memiliki memory dan status pipeline
    if user_id not in user_active_pipeline:
        user_active_pipeline[user_id] = None
    if user_id not in user_booking_memory:
        user_booking_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
    if user_id not in user_global_memory:
        user_global_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")

    # Jika user sudah berada dalam pipeline booking, langsung proses booking
    if user_active_pipeline[user_id] == "booking":
        booking_chain_instance = LLMChain(llm=llm, prompt=booking_prompt, memory=user_booking_memory[user_id])
        response = booking_chain_instance.run(user_query=message)
        # Contoh: jika respons mengandung "booking dikonfirmasi", reset status pipeline
        if "booking dikonfirmasi" in response.lower():
            user_active_pipeline[user_id] = None
    else:
        # Lakukan intent classification
        intent = intent_chain.run(user_query=message).strip().lower()
        print(f"User {user_id} detected intent: {intent}")
        if intent == "booking":
            user_active_pipeline[user_id] = "booking"
            booking_chain_instance = LLMChain(llm=llm, prompt=booking_prompt, memory=user_booking_memory[user_id])
            response = booking_chain_instance.run(user_query=message)
        elif intent == "information":
            response = doctor_chain.run(doctor_data=doctor_data_str, user_query=message)
        elif intent == "greetings":
            response = "Halo, selamat datang di layanan dokter kami! Bagaimana saya dapat membantu Anda hari ini?"
        elif intent == "goodbye":
            response = "Terima kasih telah menghubungi kami. Semoga hari Anda menyenangkan!"
        else:
            response = "Maaf, pesan Anda tidak terkait dengan layanan dokter kami. Silakan ajukan pertanyaan lain yang relevan."

    # Simpan riwayat percakapan global untuk user
    user_global_memory[user_id].save_context({"user_query": message}, {"response": response})
    return response

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    response = process_user_query(chat_request.user_id, chat_request.message)
    return ChatResponse(response=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
