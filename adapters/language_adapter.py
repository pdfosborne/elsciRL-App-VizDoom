from transformers import pipeline

class LanguageAdapter:
    def __init__(self):
        self.llm = pipeline('text-generation', model="gpt-3.5-turbo", max_length=128)
        self.obs_mapping = {}

    def adapter(self, state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        img = state.screen_buffer
        pixel_summary = f"Screen buffer shape: {img.shape}, mean pixel: {img.mean():.2f}"

        prompt = (
            "You are playing the original Doom game, describe with as much detail as possible "
            "the current position in the game that would be used to make actions and react to challenges.\n\n"
            f"Image details: {pixel_summary}\nDescription:"
        )

        if not encode:
            return prompt

        response = self.llm(prompt)[0]['generated_text']
        state_text = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

        if indexed:
            if state_text not in self.obs_mapping:
                self.obs_mapping[state_text] = len(self.obs_mapping)
            return self.obs_mapping[state_text]

        # For further encoding you may hook this up to a language embedding model (e.g., sentence-transformers)
        return state_text
