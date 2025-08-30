import os
import glob
import re
from langchain_community.document_loaders import UnstructuredPDFLoader


# --- Configuración ---
SOURCE_DIRECTORY = "data"       # Directorio donde están tus PDFs originales
OUTPUT_DIRECTORY = "processed_data" # Directorio donde guardaremos los .txt limpios

# --- Funciones de Limpieza ---

def clean_text(text):
    """
    Aplica una serie de reglas de limpieza al texto extraído.
    Esta función es el corazón de nuestro pipeline.
    """
    # 1. Eliminar cabeceras y pies de página comunes de Microsoft Learn
    # Ejemplo: "8/28/25, 1:23 AM What is Azure OpenAI... | Microsoft Learn 6/6"
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M .+? \| Microsoft Learn \d+/\d+', '', text)
    
    # 2. Eliminar patrones de "Página X de Y"
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    
    # 3. Eliminar URLs completas (pueden confundir al LLM)
    # Dejamos los nombres de dominio si son parte del texto.
    text = re.sub(r'https?://\S+', '', text)

    # 4. Eliminar saltos de línea múltiples y espacios extra
    # Reemplaza 3 o más saltos de línea con solo 2 (deja un párrafo de espacio)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Reemplaza espacios múltiples con un solo espacio
    text = re.sub(r' {2,}', ' ', text)
    # Elimina espacios al principio y final de cada línea
    text = "\n".join([line.strip() for line in text.split('\n')])

    # --- Puedes añadir más reglas de limpieza aquí según veas patrones en tus datos ---
    # Ejemplo: Eliminar líneas de copyright
    # text = re.sub(r'© \d{4} Microsoft Corporation. All rights reserved.', '', text)

    return text.strip()


# --- Proceso Principal ---

def main():
    """
    Función principal que orquesta el pipeline de pre-procesamiento FORZANDO OCR.
    """
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Patrón 1: Busca PDFs en el directorio raíz (ej: data/*.pdf)
    root_level_pdfs = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.pdf"))
    
    # Patrón 2: Busca PDFs en todos los subdirectorios (ej: data/azure/*.pdf)
    sub_dir_pdfs = glob.glob(os.path.join(SOURCE_DIRECTORY, "**/*.pdf"), recursive=True)
    
    # Unimos las dos listas y usamos un set para eliminar duplicados,
    # en caso de que el segundo patrón también capture los del raíz.
    pdf_files = sorted(list(set(root_level_pdfs + sub_dir_pdfs)))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en el directorio '{SOURCE_DIRECTORY}'.")
        return

    print(f"Se encontraron {len(pdf_files)} archivos PDF para procesar con OCR forzado.")

    for pdf_path in pdf_files:
        print(f"\n--- Procesando con OCR: {pdf_path} ---")
        try:
            # Le decimos al loader que use la estrategia "hi_res", que fuerza el OCR
            # y es ideal para documentos con layouts complejos o escaneados.
            loader = UnstructuredPDFLoader(pdf_path, strategy="hi_res")
            pages = loader.load()
            
            full_text = "\n".join([doc.page_content for doc in pages])
            
            print("  Limpiando texto extraído con OCR...")
            cleaned_text = clean_text(full_text)
            
            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            output_filename = f"{base_filename}.txt"
            output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"  ¡Éxito! Texto de OCR limpio guardado en: {output_path}")

        except Exception as e:
            print(f"  *** ERROR procesando el archivo {pdf_path}: {e} ***")

    print("\n¡Pipeline de pre-procesamiento completado!")

if __name__ == "__main__":
    main()