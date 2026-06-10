import google.generativeai as genai
import os

# Configura tu API Key (debes obtenerla en Google AI Studio)
genai.configure(api_key="TU_API_KEY_AQUI")
model = genai.GenerativeModel('gemini-1.5-pro')

def ask_about_code(prompt, file_path=None):
    context = ""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            context = f.read()
    
    full_prompt = f"Contexto del archivo:\n{context}\n\nPregunta: {prompt}"
    response = model.generate_content(full_prompt)
    print(response.text)

# Lógica simple para recibir argumentos desde la consola
if __name__ == "__main__":
    import sys
    # Ejemplo: python ask_gemini.py "Explica este código" "src/core/dataio.py"
    ask_about_code(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)