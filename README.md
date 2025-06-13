
# Raspberry RAG: Um Pipeline de RAG Otimizado para Raspberry Pi

Este projeto implementa um pipeline completo de **Retrieval-Augmented Generation (RAG)** projetado para rodar de forma eficiente em hardware com recursos limitados, como um Raspberry Pi. Ele permite que você faça perguntas em linguagem natural sobre uma coleção de documentos PDF e receba respostas geradas por um modelo de linguagem local (LLM) rodando via Ollama.

O sistema é composto por duas partes principais:
1.  **Processador de PDFs (`pdf_processor.py`):** Uma ferramenta de linha de comando que lê uma pasta de documentos PDF, os divide em pedaços, gera embeddings vetoriais e os armazena em um banco de dados vetorial customizado e leve.
2.  **API de Busca e RAG (`vector_api.py`):** Um servidor FastAPI que expõe endpoints para realizar buscas vetoriais e um pipeline RAG completo, desde a consulta até a geração da resposta em streaming.

## ✨ Funcionalidades

-   **Processamento de PDFs:** Extrai texto de múltiplos arquivos PDF e os divide em pedaços (chunks) gerenciáveis.
-   **Embeddings Locais:** Utiliza `sentence-transformers` para gerar embeddings vetoriais localmente, sem depender de APIs externas.
-   **Banco de Dados Vetorial Leve:** Implementa um `HierarchicalVectorTape`, um sistema de armazenamento customizado que combina:
    -   **Filtro de Bloom:** Para uma pré-filtragem extremamente rápida de documentos relevantes.
    -   **Quantização de Vetores:** Para reduzir o uso de disco e memória.
    -   **Arquivos Mapeados em Memória:** Para buscas eficientes com baixo overhead.
-   **API com FastAPI:** Oferece uma API robusta e assíncrona para consultas.
-   **Integração com Ollama:** Conecta-se a um servidor Ollama local para geração de texto, permitindo o uso de diversos LLMs open-source (Llama3, Phi-3, etc.).
-   **Streaming de Respostas:** As respostas do LLM são transmitidas em tempo real (Server-Sent Events), proporcionando uma experiência de usuário mais interativa e responsiva.
-   **Otimizado para Edge/IoT:** Projetado com baixo consumo de recursos para rodar em dispositivos como o Raspberry Pi.

## 🏛️ Arquitetura

  <!-- Você pode criar um diagrama e subir no imgur ou similar para colocar aqui -->

1.  **Indexação (Offline):**
    -   O `pdf_processor.py` é executado.
    -   Ele lê os PDFs da pasta `/pdfs`.
    -   Os textos são divididos e tokenizados.
    -   O modelo `all-MiniLM-L6-v2` gera embeddings para cada pedaço de texto.
    -   Os tokens são adicionados a um Filtro de Bloom, e os vetores (quantizados) são salvos em um arquivo "tape". Metadados são salvos separadamente.
2.  **Consulta (Online):**
    -   O usuário envia uma `query` para o endpoint `/rag/query` da API.
    -   A API gera o embedding da query.
    -   O Filtro de Bloom rapidamente descarta documentos que não contêm os tokens da query.
    -   Uma busca por similaridade de cosseno é realizada nos vetores candidatos.
    -   Os `top-k` trechos de texto mais relevantes são recuperados.
    -   Um prompt é construído, contendo o contexto recuperado e a pergunta original do usuário.
    -   O prompt é enviado para o modelo LLM rodando no Ollama.
    -   A API transmite a resposta do LLM de volta para o cliente, token por token.

## 🚀 Como Usar

### Pré-requisitos

-   Python 3.9+
-   [Ollama](https://ollama.com/) instalado e rodando.
-   Um modelo LLM baixado via Ollama (ex: `ollama pull phi3:mini`).
-   Um ambiente virtual Python (recomendado).

### 1. Instalação

Primeiro, clone o repositório:
```bash
git clone https://github.com/seu-usuario/raspberry-rag.git
cd raspberry-rag
```

Crie e ative um ambiente virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:
> **Nota para Raspberry Pi (ARM):** O PyTorch (`torch`) precisa ser instalado manualmente primeiro. Consulte o [guia oficial do PyTorch](https://pytorch.org/) para a wheel correta para sua arquitetura. Após instalar o torch, o restante pode ser instalado via `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 2. Adicione seus Documentos

Coloque todos os seus arquivos `.pdf` dentro da pasta `/pdfs`.

### 3. Indexe os Documentos

Execute o script de processamento para criar o banco de dados vetorial. Isso só precisa ser feito uma vez ou sempre que você adicionar/modificar seus PDFs.

```bash
python pdf_processor.py
```
Ao final, uma pasta `/vector_db` será criada com os dados indexados.

### 4. Inicie a API

Inicie o servidor FastAPI. Você pode configurar o modelo Ollama a ser usado através de uma variável de ambiente.

```bash
# Exemplo usando o modelo 'phi3:mini'
export OLLAMA_MODEL="phi3:mini"

# Inicia o servidor
python vector_api.py
```
A API estará rodando em `http://localhost:8000`.

### 5. Faça uma Consulta

Você pode usar `curl` ou qualquer cliente de API para fazer uma pergunta.

```bash
curl -X POST "http://localhost:8000/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Faça um resumo dos pdfs"}'
```

A resposta será transmitida em formato Server-Sent Events (SSE):
```
data: {"type": "sources", "data": [{"source": "A_tutorial_on_stochastic_programming.pdf", "chunk_id": 2, "similarity": 0.2173}]}

data: {"type": "llm_token", "data": "A"}

data: {"type": "llm_token", "data": " principal"}

data: {"type": "llm_token", "data": " conclus\u00e3o"}

data: {"type": "llm_token", "data": " do"}

data: {"type": "llm_token", "data": " paper"}

...
```

## 🛠️ Configuração e Implantação

### Variáveis de Ambiente

O comportamento da API pode ser controlado por variáveis de ambiente:
-   `OLLAMA_MODEL`: Nome do modelo a ser usado pelo Ollama (padrão: `gemma3:1b`).
-   `OLLAMA_HOST`: URL do servidor Ollama (padrão: `http://localhost:11434`).
-   `HOST`: Host em que a API irá escutar (padrão: `0.0.0.0`).
-   `PORT`: Porta da API (padrão: `8000`).

### Implantação no Raspberry Pi

1.  **Instale o Ollama** no Pi.
2.  **Clone o repositório** e instale as dependências (lembre-se de instalar o `torch` específico para ARM).
3.  **Recrie o banco de dados** executando `pdf_processor.py` diretamente no Pi, pois os arquivos binários podem não ser portáteis entre arquiteturas.
4.  **Inicie a API** usando um script ou serviço (veja `start_api.sh` como exemplo).
5.  **(Opcional) Exponha para a Internet** usando um serviço de túnel como o [Ngrok](https://ngrok.com/): `ngrok http 8000`.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request para melhorias, correções de bugs ou novas funcionalidades.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
```