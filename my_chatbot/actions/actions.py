# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionProvideDiagnosis(Action):
    def name(self) -> Text:
        return "action_provide_diagnosis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_input = tracker.latest_message.get("text").lower()

        if "chest pain" in user_input and "breathing" in user_input:
            diagnosis = "This could be a serious condition like a heart issue or asthma. Please seek immediate medical attention."
        elif "headache" in user_input and "fever" in user_input:
            diagnosis = "You may have the flu or a viral infection. Drink fluids and rest, but consult a doctor to be sure."
        elif "sore throat" in user_input and "cough" in user_input:
            diagnosis = "These are common symptoms of a cold or flu."
        else:
            diagnosis = "I'm not sure what it could be. It’s best to consult a doctor."

        dispatcher.utter_message(text=diagnosis)
        return []


class ActionProvideTreatment(Action):
    def name(self) -> Text:
        return "action_provide_treatment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_input = tracker.latest_message.get("text").lower()
        if "headache" in user_input:
            treatment = "You can take paracetamol and rest in a quiet, dark room."
        elif "fever" in user_input:
            treatment = "Stay hydrated and take fever-reducing medication like paracetamol."
        elif "cough" in user_input:
            treatment = "Try warm fluids, throat lozenges, and a humidifier. If it persists, see a doctor."
        else:
            treatment = "It’s best to consult a doctor for proper medication."

        dispatcher.utter_message(text=treatment)
        return []


class ActionProvidePrevention(Action):
    def name(self) -> Text:
        return "action_provide_prevention"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        prevention_tips = (
            "To prevent common illnesses:\n"
            "- Wash your hands regularly\n"
            "- Eat a balanced diet\n"
            "- Exercise regularly\n"
            "- Get enough sleep\n"
            "- Avoid close contact with sick individuals"
        )
        dispatcher.utter_message(text=prevention_tips)
        return []


class ActionProvideConditionInfo(Action):
    def name(self) -> Text:
        return "action_provide_condition_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_input = tracker.latest_message.get("text").lower()

        if "diabetes" in user_input:
            info = "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision."
        elif "heart disease" in user_input:
            info = "Signs of heart disease may include chest pain, shortness of breath, and fatigue."
        elif "stroke" in user_input:
            info = "Warning signs of a stroke include sudden numbness, confusion, trouble speaking, and dizziness."
        elif "asthma" in user_input:
            info = "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing."
        else:
            info = "I can help with general info, but please consult a professional for an accurate understanding."

        dispatcher.utter_message(text=info)
        return []


class ActionProvideGeneralHealth(Action):
    def name(self) -> Text:
        return "action_provide_general_health"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        health_tips = (
            "Here are some general health tips:\n"
            "- Drink at least 2 liters of water a day\n"
            "- Eat fruits and vegetables daily\n"
            "- Exercise for at least 30 minutes\n"
            "- Maintain a consistent sleep schedule"
        )
        dispatcher.utter_message(text=health_tips)
        return []


class ActionSuggestNextSteps(Action):
    def name(self) -> Text:
        return "action_suggest_next_steps"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        suggestion = (
            "Based on your symptoms, it’s best to consult a healthcare professional.\n"
            "If your condition worsens, visit a clinic or hospital immediately."
        )
        dispatcher.utter_message(text=suggestion)
        return []
