from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import json
from datetime import datetime

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

# === Global Dictionaries untuk Memory & Pipeline per User ===
user_symptom_memory = {}  # Memory untuk menyimpan gejala user
user_global_memory = {}  # Memory percakapan umum
user_active_pipeline = {}  # Menyimpan status pipeline user


def get_greeting_time():
    hour = datetime.now().hour
    if hour < 12:
        return "pagi"
    elif hour < 15:
        return "siang"
    elif hour < 18:
        return "sore"
    else:
        return "malam"


# === 2. Intent Classification Chain (Global) ===
intent_prompt_template = """
Percakapan sebelumnya:
{chat_history}

User message: {user_query}

Klasifikasikan pesan user ke dalam salah satu kategori berikut:
- symptom_check: jika user menyebutkan gejala atau memberikan respon terkait gejala.
- booking: jika user ingin membuat janji temu dengan dokter.
- information: jika user meminta informasi tentang dokter atau layanan.
- greetings: jika user menyapa.
- goodbye: jika user mengucapkan selamat tinggal.
- out of topic: jika pesan tidak terkait dengan layanan medis.

Jika user hanya menjawab dengan satu kata seperti "ya", "tidak", "mungkin", tetap anggap sebagai bagian dari "symptom_check".

Jawab hanya dengan salah satu kategori di atas.
"""
intent_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=intent_prompt_template
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

greetings_prompt_template = """
Anda adalah asisten AI yang ramah. Berdasarkan pesan user, buatlah salam yang lebih personal.

**Riwayat percakapan sebelumnya:**
{chat_history}

**Pesan terbaru dari user:**
User: {user_query}

**Instruksi:**
1. Jika user menyebutkan nama mereka (misalnya: "Halo, saya Faridan"), ingatlah nama tersebut dan gunakan dalam respons Anda.
2. Jika user menyebutkan waktu (pagi/siang/sore/malam), sesuaikan salam dengan waktu tersebut.
3. Gunakan gaya bahasa yang ramah dan profesional.

**Jawaban harus dalam format berikut:**
- Jika user menyebutkan nama: "Halo, [Nama]! Selamat [waktu], bagaimana saya bisa membantu Anda hari ini?"
- Jika user tidak menyebutkan nama: "Selamat [waktu]! Bagaimana saya bisa membantu Anda hari ini?"
"""


# Prompt untuk model LLM
greetings_prompt_template = """
Anda adalah asisten AI yang ramah. Berdasarkan pesan user, buatlah salam yang langsung dan natural.

**Pesan terbaru dari user:**
User: {user_query}

**Jawaban yang diharapkan:**
- Jika user menyebutkan nama: "Halo, [Nama]! Selamat {greeting_time}, bagaimana saya bisa membantu Anda hari ini?"
- Jika user tidak menyebutkan nama: "Selamat {greeting_time}! Bagaimana saya bisa membantu Anda hari ini?"
"""
greetings_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "greeting_time"],
    template=greetings_prompt_template
)

goodbye_prompt_template = """
Anda adalah asisten AI yang ramah. Berdasarkan pesan user, buatlah ucapan perpisahan yang lebih personal.

**Riwayat percakapan sebelumnya:**
{chat_history}

**Pesan terbaru dari user:**
User: {user_query}

**Instruksi:**
1. Jika user menyebutkan nama, gunakan nama tersebut dalam salam perpisahan.
2. Gunakan gaya bahasa yang ramah dan profesional.

**Jawaban harus dalam format berikut:**
- Jika user menyebutkan nama: "Terima kasih, [Nama]! Semoga hari Anda menyenangkan. Sampai jumpa lagi!"
- Jika user tidak menyebutkan nama: "Terima kasih telah menghubungi kami. Semoga sehat selalu! Sampai jumpa!"
"""
goodbye_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=goodbye_prompt_template
)

# === 3. Dummy Data Dokter ===
def load_doctors():
    with open("doctors.json", "r", encoding="utf-8") as file:
        return json.load(file)

doctors = load_doctors()

doctor_data_str = "\n".join([
    f"Name: {doc['name']}, Specialist: {doc['specialist']}, Schedule: {doc['schedule']}"
    for doc in doctors
])

# === 4. Prompt Diagnosa Gejala dengan Follow-up Singkat ===
diagnosis_prompt_template = f"""
Anda adalah asisten medis AI. Analisis gejala user secara bertahap sebelum memberikan kesimpulan.

**Riwayat percakapan sebelumnya:**
{{chat_history}}

**Pesan terbaru:**
User: {{user_query}}

**Instruksi:**
1. Jika ini adalah pertanyaan pertama user tentang gejala, JANGAN langsung memberikan diagnosis atau rekomendasi dokter. Sebagai gantinya, ajukan satu pertanyaan tambahan untuk memahami lebih lanjut gejalanya.
2. Jika user sudah memberikan jawaban tambahan atau gejala cukup jelas, baru berikan kemungkinan penyakit dan spesialisasi dokter yang sesuai.
3. Hindari memberikan saran medis final, hanya rekomendasi awal.

**Jawaban harus dalam format berikut:**
- Kemungkinan penyakit: [sebutkan kemungkinan penyakit]
- Spesialis yang direkomendasikan: [sebutkan spesialis]
- Rekomendasi Dokter Spesialis yang ada di bawah ini berdasarkan Spesialis yang sesuai.

{doctor_data_str}

"""



diagnosis_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=diagnosis_prompt_template
)

# === 5. Booking Appointment Chain (dengan Conversation Memory) ===
booking_prompt_template = f"""
Percakapan sebelumnya:
{{chat_history}}

Anda adalah asisten booking dokter. Pastikan detail yang diperlukan lengkap:
- Nama dokter
- Tanggal & waktu

Jika user belum memberikan semua informasi, tanyakan secara singkat.

Berikut daftar dokter:
{doctor_data_str}

User: {{user_query}}

Chatbot:
"""
booking_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=booking_prompt_template
)

# === 6. Flask API ===
app = Flask(__name__)

def process_user_query(user_id: str, message: str) -> str:
    # Reset memory jika user memasukkan "/reset"
    if message.strip() == "/reset":
        user_symptom_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
        user_global_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
        user_active_pipeline[user_id] = None
        return "Memory telah direset."

    # Pastikan setiap user memiliki memory dan status pipeline
    if user_id not in user_active_pipeline:
        user_active_pipeline[user_id] = None
    if user_id not in user_symptom_memory:
        user_symptom_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")
    if user_id not in user_global_memory:
        user_global_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", input_key="user_query")

    # Jika user dalam pipeline booking, langsung proses booking
    if user_active_pipeline[user_id] == "booking":
        booking_chain_instance = LLMChain(llm=llm, prompt=booking_prompt, memory=user_global_memory[user_id])
        response = booking_chain_instance.run(user_query=message)
        if "booking dikonfirmasi" in response.lower():
            user_active_pipeline[user_id] = None  # Reset pipeline setelah booking selesai
    else:
        # Deteksi intent user
        intent = intent_chain.run(chat_history=user_symptom_memory[user_id].buffer,user_query=message).strip().lower()
        print(f"User {user_id} detected intent: {intent}")

        if intent == "symptom_check":
            # Simpan gejala dalam memory
            user_symptom_memory[user_id].save_context({"user_query": message}, {"response": ""})

            # Cek apakah ini pertama kali user menyebutkan gejala
            previous_messages = user_symptom_memory[user_id].buffer
            if len(previous_messages) == 0:
                # Jika tidak ada riwayat sebelumnya, chatbot harus bertanya dulu
                diagnosis_prompt_initial = """
                Anda adalah asisten medis AI. User baru pertama kali menyebutkan gejala. Jangan langsung berikan diagnosis atau rekomendasi dokter.
                
                **Riwayat percakapan sebelumnya:**
                {chat_history}
                
                **Pesan terbaru:**
                User: {user_query}

                **Instruksi:**
                - Ajukan satu pertanyaan tambahan untuk memahami lebih lanjut gejalanya.
                - Hindari memberikan kesimpulan di tahap ini.
                """
                initial_diagnosis_prompt = PromptTemplate(
                    input_variables=["chat_history", "user_query"],
                    template=diagnosis_prompt_initial
                )
                diagnosis_chain_instance = LLMChain(llm=llm, prompt=initial_diagnosis_prompt, memory=user_symptom_memory[user_id])
                response = diagnosis_chain_instance.run(chat_history=user_symptom_memory[user_id].buffer, user_query=message)
            
            else:
                # Jika sudah ada percakapan sebelumnya, lanjutkan ke diagnosis
                diagnosis_chain_instance = LLMChain(llm=llm, prompt=diagnosis_prompt, memory=user_symptom_memory[user_id])
                response = diagnosis_chain_instance.run(
                    chat_history=user_symptom_memory[user_id].buffer, 
                    user_query=message
                )

        elif intent == "booking":
            user_active_pipeline[user_id] = "booking"
            booking_chain_instance = LLMChain(llm=llm, prompt=booking_prompt, memory=user_global_memory[user_id])
            response = booking_chain_instance.run(user_query=message)

        elif intent == "information":
            response = f"Berikut daftar dokter yang tersedia:\n{doctor_data_str}"

        elif intent == "greetings":
            # Gunakan LLM untuk membuat salam yang lebih personal
            greeting_time = get_greeting_time()
            greetings_chain_instance = LLMChain(llm=llm, prompt=greetings_prompt, memory=user_global_memory[user_id])
            response = greetings_chain_instance.run(chat_history=user_global_memory[user_id].buffer, user_query=message, greeting_time=greeting_time)

            # Cek apakah user menyebutkan nama mereka
            for word in message.split():
                if word.lower() in ["saya", "nama", "aku"]:
                    user_name = message.split()[-1]  # Ambil kata terakhir sebagai nama (bisa ditingkatkan dengan NLP)
                    user_global_memory[user_id].save_context({"user_query": message}, {"response": f"Nama user: {user_name}"})
                    response = response.replace("[Nama]", user_name)  # Gunakan nama dalam respons

        elif intent == "goodbye":
            # Gunakan LLM untuk membuat ucapan perpisahan yang lebih personal
            goodbye_chain_instance = LLMChain(llm=llm, prompt=goodbye_prompt, memory=user_global_memory[user_id])
            response = goodbye_chain_instance.run(chat_history=user_global_memory[user_id].buffer, user_query=message)

            # Cek apakah user sudah memberikan nama sebelumnya
            stored_name = user_global_memory[user_id].buffer
            if "Nama user:" in stored_name:
                user_name = stored_name.split("Nama user:")[-1].strip()
                response = response.replace("[Nama]", user_name)


        else:
            response = "Maaf, saya tidak mengerti. Bisa jelaskan lebih lanjut?"

    # Simpan riwayat percakapan user
    user_global_memory[user_id].save_context({"user_query": message}, {"response": response})
    return response

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"response": "user_id dan message harus disertakan."}), 400

    response = process_user_query(user_id, message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
