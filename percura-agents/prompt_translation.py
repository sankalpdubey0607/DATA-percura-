"""
prompt_translation.py
=====================
Translation layer for the Nvidia simulation engine.
Maps numeric behavioral parameters (0-100 scales) from the database
into actionable LLM system prompt instructions to guide persona behavior.
"""
import json

class PersonaPromptBuilder:
    def __init__(self, persona_row):
        """Initialize with a row dict from the sqlite database."""
        self.p = persona_row
        
    def get_attention_instruction(self):
        att = float(self.p.get('attention_budget', 45))
        if att < 20:
            return "You are highly impatient. You skim text and will abandon the task if it requires reading more than 2 sentences or clicking more than 2 buttons."
        elif att < 50:
            return "You have moderate patience. You will read essential information but will get frustrated if a process feels unnecessarily long."
        else:
            return "You are patient and thorough. You read instructions carefully and are willing to spend time navigating complex menus."
            
    def get_trust_instruction(self):
        trust = float(self.p.get('trust_prior', 0.5))
        if trust < 0.3:
            return "You are highly skeptical of digital platforms. You fear scams, hidden fees, and data theft. You demand explicit reassurance and verification before proceeding."
        elif trust < 0.6:
            return "You are cautiously optimistic but remain vigilant. You prefer known brands and social proof before making commitments."
        else:
            return "You are digitally trusting. You assume the platform works as intended and are comfortable sharing data or making payments quickly."
            
    def get_effort_instruction(self):
        effort = float(self.p.get('effort_tolerance', 5))
        if effort < 3:
            return "You have low effort tolerance. If you encounter an error or confusing UI, you immediately give up rather than trying to fix it."
        elif effort < 7:
            return "You are willing to put in moderate effort. You will try one or two workarounds if you encounter friction, but will abandon if blocked repeatedly."
        else:
            return "You are resilient. You will actively try to solve problems, search for help, or try multiple approaches to achieve your goal."
            
    def get_friction_instruction(self):
        triggers = self.p.get('top_friction_triggers', '[]')
        try:
            triggers_list = json.loads(triggers)
        except:
            triggers_list = []
            
        if not triggers_list:
            return ""
            
        return f"You are particularly sensitive to these specific issues: {', '.join(triggers_list)}. If you encounter them, your frustration increases exponentially."
        
    def build_system_prompt(self):
        """Constructs the complete behavioral system prompt for the LLM."""
        instructions = [
            f"You are playing the persona of: {self.p.get('extracted_name', 'A user')}, a {self.p.get('age', '30')} year old {self.p.get('occupation', 'worker')} from {self.p.get('district', 'India')}.",
            "BEHAVIORAL CONSTRAINTS:",
            self.get_attention_instruction(),
            self.get_trust_instruction(),
            self.get_effort_instruction(),
            self.get_friction_instruction()
        ]
        
        # Remove empty lines
        return "\n".join([line for line in instructions if line.strip()])

# Example usage
if __name__ == "__main__":
    sample_row = {
        "extracted_name": "Ramesh",
        "age": 45,
        "occupation": "Farmer",
        "district": "Pune Rural",
        "attention_budget": 15,
        "trust_prior": 0.2,
        "effort_tolerance": 2,
        "top_friction_triggers": '["complex_ui", "hidden_fees"]'
    }
    
    builder = PersonaPromptBuilder(sample_row)
    print("=== SAMPLE SYSTEM PROMPT ===")
    print(builder.build_system_prompt())
