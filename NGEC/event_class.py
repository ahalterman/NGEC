from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
# a safer pickle alternative
import skops.io as sio
import numpy as np


class EventClass:
    def __init__(self, 
                 model_dir="NGEC/assets/event_models/",
                 threshold=0.6, 
                 progress_bar=False,
                 event_types = ['ACCUSE', 'AGREE', 'AID', 'ASSAULT', 'COERCE', 'CONCEDE',
                                'CONSULT', 'COOPERATE', 'MOBILIZE', 'PROTEST', 'REJECT', 'REQUEST',
                                'RETREAT', 'SANCTION', 'SUPPORT', 'THREATEN']
                 ):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
        self.model_dir = model_dir
        self.threshold = threshold
        self.progress_bar = progress_bar
        self.event_types = event_types
        self.model_dict = self._load_model(model_dir)
        print("Event classification models loaded. NOTE: these models are not the production models used to produce the POLECAT dataset. Instead, these are demonstration models for the PLOVER ontology trained on synthetic text. If you are making custom event data, you'll need to train your own models. See the `setup` directory in the NGEC repo (github.com/ahalterman/NGEC)`.")
    
    def _load_model(self, model_dir):
        """
        Load the event classification models. 

        Parameters
        ----------
        model_dir: Path
          path to the event classification models

        Returns
        ------
        model_dict: dict
          With event classes as keys and models as values.
        """
        model_dict = {}
        for event in self.event_types:
            model_dict[event] = sio.load(f"{model_dir}/{event}.skops")
        return model_dict

    def process(self, story_list):
        """
        Process a list of stories to detect the event class.
        
        Example
        -------
        The input is a list of dictionaries, each with an 'event_text' key with the full text of the story.

        {'date': '2019-08-01',
         'event_text': 'Indonesia is investigating a report that ... ',
         'headline': 'Indonesia says it is probing a report of a ...',
         'id': '<internal document id>',
         'pub_date': '2019-08-01',
         'publisher': '<publisher name>',
         'story_id': '<recommended ID/url of original text>',
         'version': '<optional version number>'} 

        Parameters
        ----------
        story_list: list of dicts
          Each dictionary must have a 'text' key with the full text of the story.
        
        Returns
        -------
        story_list: list of dicts
          Each story dictionary now contains an 'event_type' key with a list of detected events (str). E.g.: 
          'event_type': ['SANCTION', 'MOBILIZE']

        """
        text = [i['event_text'] for i in story_list]
        embeddings = self.model.encode(text, show_progress_bar=self.progress_bar)

        preds = []
        for event, clf in self.model_dict.items():
            y_pred = clf.predict_proba(embeddings)[:,1]
            preds.append(y_pred)

        pred_array = np.array(preds).T

        # convert the matrix of predictions to a list of lists
        preds = []
        for i in pred_array:
            preds.append([self.event_types[j] for j in np.where(i > self.threshold)[0]])

        for n, story in enumerate(story_list):
            story['event_type'] = preds[n]

        return story_list
