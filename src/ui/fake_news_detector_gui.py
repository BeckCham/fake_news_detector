from src.models import predict_from_url


class FakeNewsDetectorGUI:

    def __init__(self, url_entry, is_box_ticked, canvas, result_title, result_text,result_title_2,result_title_3):
        self.url_entry = url_entry
        self.is_box_ticked = is_box_ticked
        self.canvas = canvas
        self.result_title_id = result_title
        self.result_text_id = result_text
        self.result_title_2_id = result_title_2
        self.result_title_3_id = result_title_3

    def classify_url(self):
        """Called when classify button is pressed"""
        url = self.url_entry.get()
        canvas = self.canvas
        print("button")

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

        """
        classifier_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'clickbait', 'unreliable',
                                 'political', 'reliable']
        """
        prediction = predict_from_url.predict(url)

        detailed = self.is_box_ticked.get()

        if prediction is None:
            canvas.itemconfig(self.result_title_id, text="URL cannot be read")
            canvas.itemconfig(self.result_text_id, text="Please check the URL you entered was correct and try again")
        else:
            classification = prediction[0]['label']
            certainty = prediction[0]['confidence']
            canvas.itemconfig(self.result_title_id, text=classification)
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
            result_title = label_to_title[classification]
            if detailed:
                second_classification = label_to_title[prediction[1]['label']]
                second_certainty = prediction[1]['confidence']
                third_classification = label_to_title[prediction[2]['label']]
                third_certainty = prediction[2]['confidence']
                canvas.itemconfig(self.result_title_id, text=(f'{result_title}: {certainty*100:.2f}%'))
                canvas.itemconfig(self.result_title_2_id, text=(f'{second_classification}: {second_certainty*100:.2f}%'))
                canvas.itemconfig(self.result_title_3_id, text=(f'{third_classification}: {third_certainty*100:.2f}%'))
            else:
                #Applies the classification
                canvas.itemconfig(self.result_title_id, text=result_title)
                canvas.itemconfig(self.result_text_id, text=result_text)





