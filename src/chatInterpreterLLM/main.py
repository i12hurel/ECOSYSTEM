import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

from src.chatInterpreterLLM.env import enviroment
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text

import json
import time
import re

API_KEY = enviroment()

info_text = return_info_text()
info_experto = return_info_experto()

from src.chatInterpreterLLM.crew.LIME_crew import return_resultado_crew_LIME
result_crew_LIME = return_resultado_crew_LIME()
print("CREW LIME FINISHED")

from src.chatInterpreterLLM.crew.SHAP_crew import return_resultado_crew_SHAP
result_crew_SHAP = return_resultado_crew_SHAP()
print("CREW SHAP FINISHED")

from src.chatInterpreterLLM.crew.ANCHOR_crew import return_resultado_crew_ANCHOR
result_crew_ANCHOR = return_resultado_crew_ANCHOR()
print("CREW ANCHOR FINISHED")

from src.chatInterpreterLLM.crew.results_crew import return_resultado_crew_integrador
result_crew_integrador = return_resultado_crew_integrador()

print("CREW INTEGRADOR FINISHED")
#print("INTEGRADOR RESULT: ", result_crew_integrador)

with open("resultado_integrador.txt", "w", encoding="utf-8") as f:
    f.write(f"{result_crew_integrador}\n")