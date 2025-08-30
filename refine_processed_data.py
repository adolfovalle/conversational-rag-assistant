import os
import glob
import re

def clean_text(text):
    """
    Aplica una serie de reglas de limpieza modulares y más robustas al texto extraído.
    """
    # 1. Eliminar fechas y horas en formato "M/D/YY, H:MM AM/PM"
    # Esta regla busca la fecha, la coma, la hora y el AM/PM.
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s+[AP]M', '', text)

    # 2. Eliminar números de página en formato "X/Y" que están solos en una línea
    # ^ y $ con re.MULTILINE aseguran que solo borremos líneas que CONTIENEN ÚNICAMENTE el número de página.
    text = re.sub(r'^\d{1,2}\s*/\s*\d{1,2}$', '', text, flags=re.MULTILINE)

    # 3. Eliminar texto de banners y pop-ups (basado en tus ejemplos anteriores)
    text = re.sub(r'Build Your Agile Team.*?Your message\.', '', text, flags=re.DOTALL)
    text = re.sub(r'Join \d+ Subscribers.*?Cookie Policy\.\s*Accept x', '', text, flags=re.DOTALL)
    text = re.sub(r'We\'re Online! How may I help you tod\.\.\.', '', text)
    text = re.sub(r'METADESIGN SOLUTIONS\s*@\)', '', text)
    text = re.sub(r'WPWEB INFOTECH', '', text)
    
    # 4. Eliminar lo que parece ser un embedding de OpenAI o datos JSON
    text = re.sub(r'\[\s*@\.\d+,.*?\]', '', text, flags=re.DOTALL)

    # 5. Limpieza general de espacios y saltos de línea
    # Reemplaza 3 o más saltos de línea con solo 2 (deja un párrafo de espacio)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Elimina cualquier línea que haya quedado completamente vacía (solo espacios)
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)
    # Reemplaza espacios múltiples con un solo espacio
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()

def main():
    SOURCE_DIRECTORY = "processed_data"
    
    txt_files = glob.glob(os.path.join(SOURCE_DIRECTORY, "**/*.txt"), recursive=True)

    if not txt_files:
        print(f"No se encontraron archivos .txt en '{SOURCE_DIRECTORY}'.")
        print("Asegúrate de haber ejecutado 'preprocess.py' primero.")
        return

    print(f"Refinando {len(txt_files)} archivos de texto pre-procesados...")

    for txt_path in txt_files:
        print(f"  Refinando: {txt_path}")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            refined_text = clean_text(original_text)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(refined_text)
        
        except Exception as e:
            print(f"    *** ERROR refinando el archivo {txt_path}: {e} ***")

    print("\n¡Refinamiento de datos completado!")

if __name__ == "__main__":
    main()