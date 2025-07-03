# ECOSYSTEM-MODULADO

Este proyecto contiene un sistema de interpretación de modelos de machine learning utilizando técnicas como LIME, SHAP y ANCHOR, implementado con CrewAI y otros módulos de IA explicable.

## Estructura del proyecto

```
ECOSYSTEM-MODULADO/
├── src/                          # Código fuente
│   ├── chatInterpreterLLM/
│       ├── crew/
│       ├── env/
│       ├── knowledge/
│       ├── tasks/
│       ├── tools/
│       └── training/
├── requirements.txt              # Dependencias del proyecto
├── .vscode/settings.json          # Configuración recomendada para VSCode
└── README.md                      # Instrucciones
```

## 🚀 Cómo empezar

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/ECOSYSTEM-MODULADO.git
cd ECOSYSTEM-MODULADO
```

### 2. Crea un entorno virtual

```bash
python -m venv venv
```

### 3. Activa el entorno virtual

* En **Windows**:

  ```bash
  venv\Scripts\activate
  ```
* En **Mac/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 4. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 5. Configura Visual Studio Code (VSCode)

#### Paso 1: Selecciona el intérprete Python

* `Ctrl+Shift+P` → **Python: Select Interpreter** → Elige el Python de `./venv`.

#### Paso 2: Añade configuración para los imports

Asegúrate de tener este archivo en `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [
    "./src"
  ],
  "python.defaultInterpreterPath": "./venv/Scripts/python.exe"
}
```

*(En Mac/Linux, la ruta es `./venv/bin/python`.)*

Esto permite que VSCode reconozca correctamente los imports relativos desde `src/`.

### 6. Ejecutar el proyecto

Desde la raíz del proyecto:

```bash
python -m src.chatInterpreterLLM.main
```

---

## ✅ Requisitos

* Python 3.10 o superior
* Visual Studio Code (opcional, pero recomendado)

---

## ⚠️ Notas importantes

* **No subas el `venv/`** a GitHub. Usa `.gitignore`.
* Asegúde tu conexión a internet para usar CrewAI con los modelos LLM.

---
