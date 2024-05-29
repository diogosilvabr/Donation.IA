
# Donation.IA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

Este projeto utiliza um modelo de machine learning para analisar textos e identificar linguagem inadequada. 
O modelo é treinado utilizando o LightGBM, árvore de decisão e dados de feedback para melhorar suas predições ao longo do tempo.

## Índice

- [Visão Geral](#visão-geral)
- [Instalação](#instalação)
- [Uso](#uso)
- [Contribuição](#contribuição)
- [Licença](#licença)

## Visão Geral

O objetivo deste projeto é criar uma API que possa ser usada para analisar textos e identificar se contêm linguagem inadequada. 
O modelo utiliza o LightGBM para treinamento e é continuamente aprimorado com dados de feedback fornecidos pelos usuários.

## Instalação

Siga estas etapas para configurar e executar o projeto localmente.

### Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes do Python)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/diogosilvabr/Donation.IA.git
   cd Donation.IA
   ```

2. Crie um ambiente virtual (opcional, mas recomendado em caso de problemas):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente (opcional):
   - Copie o arquivo `.env.example` para `.env` e configure as variáveis necessárias.
   - Exemplo de conteúdo do `.env`:
     ```
     FLASK_ENV=development
     ```

5. Execute a aplicação:
   ```bash
   flask run ou python run.py
   ```

## Uso

### Endpoints

#### Analisar Texto

- **URL**: `/analyze-text`
- **Método**: `POST`
- **Parâmetros de Entrada**: JSON com o campo `text`
- **Exemplo de Entrada**:
  ```json
  {
    "text": "Seu texto aqui"
  }
  ```
- **Exemplo de Saída**:
  ```json
  {
    "inapropriado": true
  }
  ```

#### Adicionar Feedback

- **URL**: `/add-feedback`
- **Método**: `POST`
- **Parâmetros de Entrada**: JSON com os campos `text` e `inappropriate`
- **Exemplo de Entrada**:
  ```json
  {
    "text": "Texto de exemplo",
    "inappropriate": 1
  }
  ```
- **Exemplo de Saída**:
  ```json
  {
    "message": "Feedback added successfully"
  }
  ```

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões, encontrar bugs ou quiser contribuir com o código, por favor, siga estes passos:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/sua-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/sua-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Autor

[Diogo Silva](https://github.com/diogosilvabr)

## Referências

- A base de dados [ToLD-BR](https://github.com/JAugusto97/ToLD-Br) foi utilizada neste projeto para melhorar a precisão do modelo de machine learning.
