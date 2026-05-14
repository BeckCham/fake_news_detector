"""
Filename: fake_news_detector_GUI.py
Author: Beck Chamberlain
Version: 0.03
Description: GUI controller for the Fake News Detector application. Integrates with the tkinter interface to handle
        inputs and outputs on the display.

"""
from src.ui import predict_from_url


class FakeNewsDetectorGUI:
    """
    GUI controller class for the fake news detector application, handles URL and tickbox input, as well as the display
    output that gives the classification result.
    """

    def __init__(self, url_entry, is_box_ticked, canvas, result_title, result_text, result_title_2, result_title_3):
        """
        Initialize the GUI controller with the tkinter interface elements.

        :param url_entry: Text entry element for the URL
        :param is_box_ticked: Boolean element which indicates whether detailed results are shown
        :param canvas: Canvas element for the display
        :param result_title: Text element ID for the primary classification result
        :param result_text: Text element ID for the description of the primary classification result
        :param result_title_2: Text element ID for the secondary classification result
        :param result_title_3: Text element ID for the tertiary classification result
        """
        self.url_entry = url_entry
        self.is_box_ticked = is_box_ticked
        self.canvas = canvas
        self.result_title_id = result_title
        self.result_text_id = result_text
        self.result_title_2_id = result_title_2
        self.result_title_3_id = result_title_3

    def classify_url(self):
        """
        Classifies the URL entered by the user and displays the result. Called when classify button is pressed.
        """
        # Get URL from input element
        url = self.url_entry.get()
        canvas = self.canvas

        # Map database classifier labels to appropriate user-friendly titles that can be displayed
        label_to_title = {
            'political': 'Political Bias',
            'reliable': 'Reliable',
            'unreliable': 'Proceed With Caution',
            'clickbait': 'Clickbait',
            'hate': 'Hate News',
            'junksci': 'Junk Science',
            'conspiracy': 'Conspiracy',
            'bias': 'Extreme Bias',
            'satire': 'Satire',
            'fake': 'Fake News',
        }

        # Get most confident predictions from model
        predictions = predict_from_url.predict(url)

        # Check if user desires detailed response
        detailed = self.is_box_ticked.get()

        # If the URL cannot be read or processed then an error message is displayed
        if predictions is None:
            canvas.itemconfig(self.result_title_id, text="URL cannot be read")
            canvas.itemconfig(self.result_text_id, text="Please check the URL you entered was correct and try again")
        else:
            # Gets top prediction and its certainty
            classification = predictions[0]['label']
            certainty = predictions[0]['confidence']

            # Sets description text based on the top classification
            if classification == 'reliable':
                result_text = (
                    'Based on the web page supplied, this news source appears to circulate news and information'
                    ' in a manner consistent with traditional and ethical practices in journalism')
            elif classification == 'political':
                result_text = (
                    'Based on the web page supplied, this news source appears to provide generally verifiable '
                    'information in support of certain points of view or political orientations.')
            elif classification == 'unreliable':
                result_text = (
                    'Based on the web page supplied, this news source appears to potentially be reliable but its '
                    'contents require further verification.')
            elif classification == 'clickbait':
                result_text = (
                    'Based on the web page supplied, this news source appears to provide generally credible content, but'
                    ' use exaggerated, misleading, or questionable headlines, social media descriptions, and/or images.')
            elif classification == 'hate':
                result_text = (
                    'Based on the web page supplied, this news source appears to actively promote racism, misogyny, '
                    'homophobia, and other forms of discrimination.')
            elif classification == 'junksci':
                result_text = (
                    'Based on the web page supplied, this news source appears to promote pseudoscience, metaphysics, '
                    'naturalistic fallacies, and other scientifically dubious claims.')
            elif classification == 'conspiracy':
                result_text = (
                    'Based on the web page supplied, this news source appears to promote kooky conspiracy theories based'
                    ' on supernatural or unproven claims')
            elif classification == 'bias':
                result_text = ('Based on the web page supplied, this news source appears to promote a particular point '
                               'of view and may rely on propaganda, decontextualized information, and opinions '
                               'distorted as facts.')
            elif classification == 'satire':
                result_text = ('Based on the web page supplied, this news source appears to use humor, irony, '
                               'exaggeration, ridicule, and false information to comment on current events.')
            elif classification == 'fake':
                result_text = ('Based on the web page supplied, this news source appears to entirely fabricate '
                               'information, disseminate deceptive content, or grossly distort actual news reports.')
            # Gets user-friendly title for top classification
            result_title = label_to_title[classification]
            # Displays top predictions description
            canvas.itemconfig(self.result_text_id, text=result_text)
            # Displays top three predictions if detailed checkbox is ticked
            if detailed:
                # Get second and third most likely classifications and the confidence for them
                second_classification = label_to_title[predictions[1]['label']]
                second_certainty = predictions[1]['confidence']
                third_classification = label_to_title[predictions[2]['label']]
                third_certainty = predictions[2]['confidence']
                # Display all top 3 classifications and their confidence
                canvas.itemconfig(self.result_title_id, text=(f'{result_title}: {certainty * 100:.2f}%'))
                canvas.itemconfig(self.result_title_2_id,
                                  text=(f'{second_classification}: {second_certainty * 100:.2f}%'))
                canvas.itemconfig(self.result_title_3_id,
                                  text=(f'{third_classification}: {third_certainty * 100:.2f}%'))
            else:
                # Displays top result
                canvas.itemconfig(self.result_title_id, text=result_title)
                # Ensures that extra detail elements are clear
                if self.result_title_2_id != "":
                    canvas.itemconfig(self.result_title_2_id,
                                      text=(''))
                    canvas.itemconfig(self.result_title_3_id,
                                      text=(''))
