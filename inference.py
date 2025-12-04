import torch
import sys
import argparse
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Nome da pasta onde o treino salvou o modelo (deve ser igual ao do script de treino)
MODEL_PATH = "modelo_soja_final"

class Predictor:
    def __init__(self, model_path):
        """
        Carrega o modelo treinado e o processador de imagens.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 A carregar o modelo de '{model_path}' para {self.device}...")
        
        try:
            # Carregar a estrutura e os pesos aprendidos
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval() # Modo de avaliação (desativa dropout, etc)
            
            # Carregar as regras de processamento (resize, normalização)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            print("✅ Modelo carregado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar o modelo: {e}")
            print("Dica: Verifique se a pasta do modelo existe e se o treino foi concluído.")
            sys.exit(1)

    def predict_image(self, image_path):
        """
        Faz a predição para uma única imagem.
        Retorna: (nome_classe, probabilidade)
        """
        try:
            # 1. Abrir a imagem
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Erro ao abrir o ficheiro '{image_path}': {e}")
            return None, 0.0

        # 2. Pré-processar a imagem (igual ao treino)
        inputs = self.processor(image, return_tensors="pt")
        inputs = inputs.to(self.device)

        # 3. Passar pelo modelo
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 4. Processar o resultado
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Obter a maior probabilidade
        score, label_id = torch.max(probs, dim=-1)
        
        # Converter ID para Nome (usando o config salvo no modelo)
        label_name = self.model.config.id2label[label_id.item()]
        confidence = score.item()

        return label_name, confidence

def main():
    # Configuração para rodar via linha de comandos
    parser = argparse.ArgumentParser(description="Classificador de Folhas de Soja")
    parser.add_argument("image_path", type=str, help="Caminho para a imagem que queres analisar")
    
    args = parser.parse_args()
    
    # Inicializa o preditor
    predictor = Predictor(MODEL_PATH)
    
    # Faz a previsão
    print(f"🔍 A analisar a imagem: {args.image_path} ...")
    classe, confianca = predictor.predict_image(args.image_path)
    
    if classe:
        print("-" * 30)
        print(f"🌿 Resultado: {classe.upper()}")
        print(f"📊 Confiança: {confianca:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    main()