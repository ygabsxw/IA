# Decision Trees from Scratch

Implementação educacional dos algoritmos clássicos de árvores de decisão: **ID3**, **C4.5** e **CART**.

## 🎯 Características

- **ID3**: Ganho de informação, atributos categóricos
- **C4.5**: Razão de ganho, suporte a contínuos
- **CART**: Índice de Gini, splits binários

## 📦 Instalação

### Desenvolvimento Local
```bash
# Clone o repositório
git clone https://github.com/ygabsxw/decision-trees-from-scratch
cd decision-trees-from-scratch

# Instale em modo desenvolvimento
pip install -e .
```

### Instalação via pip (quando publicado)
```bash
pip install decision-trees-from-scratch
```

## 🚀 Uso Rápido

```python
from decision_trees import ID3, C45, CART
import pandas as pd

# Carregue seus dados
df = pd.read_csv('seu_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Treine os modelos
id3_model = ID3(max_depth=10)
c45_model = C45(max_depth=10)
cart_model = CART(max_depth=10)

id3_model.fit(X, y)
c45_model.fit(X, y)
cart_model.fit(X, y)

# Faça predições
predictions = id3_model.predict(X_test)

# Visualize a árvore
print(id3_model.tree.pprint())

# Extraia regras
for rule in id3_model.extract_rules():
    print(f"- {rule}")
```

## 📊 Exemplos

### Play Tennis (Dataset Didático)
```python
from decision_trees.examples import load_play_tennis

X, y = load_play_tennis()
model = ID3()
model.fit(X, y)
print("Árvore ID3:")
print(model.tree.pprint())
```

### Titanic (Dataset Real)
```python
from decision_trees.examples import load_titanic

X_train, X_test, y_train, y_test = load_titanic()
model = C45(max_depth=8)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Acurácia: {accuracy:.3f}")
```

## 🔧 API Completa

### Parâmetros Comuns
- `max_depth`: Profundidade máxima da árvore
- `min_samples_split`: Mínimo de amostras para split
- `random_state`: Semente para reprodutibilidade

### Métodos
- `fit(X, y)`: Treina o modelo
- `predict(X)`: Faz predições
- `score(X, y)`: Calcula acurácia
- `extract_rules()`: Extrai regras legíveis

## 📁 Estrutura do Projeto

```
decision_trees/
├── __init__.py          # Exporta classes principais
├── algorithms/
│   ├── id3.py          # Implementação ID3
│   ├── c45.py          # Implementação C4.5
│   └── cart.py         # Implementação CART
├── core/
│   ├── node.py         # Classe Node
│   ├── metrics.py      # Entropy, Gini, etc.
│   └── utils.py        # Utilitários gerais
└── examples/
    ├── datasets.py     # Carregadores de dados
    └── demo.py         # Exemplos de uso
```

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte de uma atividade acadêmica sobre implementação de algoritmos de machine learning do zero. O objetivo é demonstrar:

1. **Compreensão teórica** dos algoritmos fundamentais
2. **Habilidades de implementação** sem bibliotecas prontas
3. **Análise comparativa** entre diferentes abordagens
4. **Engenharia de software** para código reutilizável

## 📖 Documentação Teórica

### ID3 (Iterative Dichotomiser 3)
- **Critério**: Ganho de Informação
- **Limitação**: Apenas atributos categóricos
- **Vantagem**: Simplicidade e interpretabilidade

### C4.5
- **Critério**: Razão de Ganho (corrige bias do ID3)
- **Melhoria**: Suporte nativo a atributos contínuos
- **Aplicação**: Datasets mistos (categóricos + contínuos)

### CART (Classification and Regression Trees)
- **Critério**: Índice de Gini
- **Característica**: Sempre splits binários
- **Vantagem**: Base para ensemble methods

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autor

**Seu Nome** - Estudante de Ciência da Computação
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- Email: seu.email@exemplo.com

## 🙏 Agradecimentos

- Professores e colegas da disciplina
- Comunidade open source
- Autores dos algoritmos clássicos: Ross Quinlan (ID3/C4.5), Breiman et al. (CART)