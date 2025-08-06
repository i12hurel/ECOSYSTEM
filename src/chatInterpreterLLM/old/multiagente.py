from crewai import Agent, Task, Crew, LLM
from src.chatInterpreterLLM.env import enviroment

API_KEY = enviroment()

# Crea el modelo base para los agentes
llm = LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.0,
            key=API_KEY
        )

# Agentes subordinados
agente_lengua = Agent(
    role="Experto en Lengua y Literatura",
    goal="Responder preguntas relacionadas con gramática, literatura y redacción",
    backstory="Es un profesor con 20 años de experiencia en lengua castellana y literatura.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

agente_matematicas = Agent(
    role="Experto en Matemáticas",
    goal="Resolver problemas matemáticos y explicar conceptos numéricos",
    backstory="Es un matemático especializado en álgebra, cálculo y lógica matemática.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

agente_historia = Agent(
    role="Experto en Historia",
    goal="Responder preguntas relacionadas con historia universal y regional",
    backstory="Es un historiador con conocimientos profundos en historia antigua y moderna.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Agente Coordinador
coordinador = Agent(
    role="Coordinador General",
    goal="Determinar si una pregunta requiere ayuda de un experto y delegarla correctamente",
    backstory=(
        "Eres un coordinador inteligente. Recibes preguntas y decides si puedes "
        "responderlas directamente o si debes pedir ayuda a un experto en lengua, matemáticas o historia. "
        "Cuando uses la herramienta de delegación, recuerda que los campos 'task', 'context' y 'coworker' "
        "deben ser siempre texto plano (string)."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm,
)

# Función que define la tarea principal
def crear_tarea(pregunta_usuario):
    return Task(
        description=f"""
        Has recibido la siguiente pregunta: "{pregunta_usuario}"

        Debes:
        - Evaluar el contenido de la pregunta.
        - Si pertenece a lengua y literatura, matemáticas o historia, delega al agente correspondiente.
        - Si no requiere delegación o es muy general, puedes responder tú mismo.

        IMPORTANTE:
        Cuando uses la herramienta 'Delegate work to coworker', debes pasar SIEMPRE texto plano:
        {{
            "task": "Describe aquí la tarea de forma simple",
            "context": "Describe aquí el contexto necesario",
            "coworker": "Experto en Lengua y Literatura / Experto en Matemáticas / Experto en Historia"
        }}
        """,
        agent=coordinador,
        expected_output="Una respuesta detallada a la pregunta del usuario.",
    )

# Función principal
def ejecutar_sistema_multiagente(pregunta):
    tarea = crear_tarea(pregunta)
    crew = Crew(
        agents=[coordinador, agente_lengua, agente_matematicas, agente_historia],
        tasks=[tarea],
        verbose=True,
    )
    try:
        resultado = crew.kickoff()
    except Exception as e:
        print(f"⚠️ Error durante la ejecución: {e}")
        resultado = "Lo siento, ocurrió un error al procesar tu pregunta."
    print("\n🔎 RESPUESTA FINAL:")
    print(resultado)

# Ejemplo de uso
if __name__ == "__main__":
    try:
        while True:
            pregunta_usuario = input("Escribe tu pregunta: ")
            ejecutar_sistema_multiagente(pregunta_usuario)
    except KeyboardInterrupt:
        print("\nSaliendo del sistema.")
