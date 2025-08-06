# ECOSYSTEM

Este proyecto tarta de la creación asistente conversacional inteligente que interpreta modelos de Machine Learning combinando LIME, SHAP y Anchor, integrando información experta y contextual para ofrecer explicaciones claras y comprensibles en una interfaz intuitiva.

## Estructura del proyecto

```
ECOSYSTEM/
├── src/                          # Código fuente
│   ├── chatInterpreterLLM/
│       ├── crew/
│       ├── agents/
│       ├── knowledge/
│       ├── tasks/
│       ├── tools/
│       ├── streamlit/
│       ├── training/
│       ├── ecosystem.py
│       ├── env.py
│       └── main.py
├── requirements.txt              # Dependencias del proyecto
├── .vscode        # Configuración recomendada para VSCode
    ├── settings.json 
    └── launch.json         
├── .env        
└── README.md                      # Instrucciones

```

## 🚀 Cómo empezar

### 1. Clona el repositorio

```bash
git clone https://github.com/i12hurel/ECOSYSTEM.git
cd ECOSYSTEM
```

### 2. Crea un entorno virtual (si ya existe, eliminalo y vuelvelo a crear)

```bash
python -m venv .venv
```

### 3. Activa el entorno virtual

* En **Windows**:

  ```bash
  .venv\Scripts\activate
  ```
* En **Mac/Linux**:

  ```bash
  source .venv/bin/activate
  ```

### 4. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 5. Configura Visual Studio Code (VSCode)

#### Paso 1: Selecciona el intérprete Python

* `Ctrl+Shift+P` → **Python: Select Interpreter** → Elige el Python de `./.venv`.

#### Paso 2: Añade una API KEY de gemini
 En la carpeta src/chatInterpreterLLM, añade un .env, donde añadas:
 " GEMINI_API_KEY ="TU_API_KEY" "

### 6. Ejecutar el proyecto

Desde la raíz del proyecto:

```bash
python -m streamlit run src/chatInterpreterLLM/streamlit/gemini_app.py
```
---

## ✅ Requisitos

* Python 3.10 o superior
* Visual Studio Code 

