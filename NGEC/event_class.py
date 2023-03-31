import sklearn

def _load_model(model_dir):
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
    raise NotImplementedError()


class EventClass:
    def __init__(self, 
                 model_dir="assets/event_class_models/",
                 threshold=0.6 # we can set stuff like this here
                 ):

        self.model_dict = _load_model(model_dir)
        self.threshold = threshold
    

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
          Each story dictionary now contains an "event_cats" key with a list of detected events (str). E.g.: 
          'event_type': ['SANCTION', 'MOBILIZE']


        """
        raise NotImplementedError()

        for story in stories:
            if event_text not in story.keys():
                raise ValueError("No 'event_text' key in input.")