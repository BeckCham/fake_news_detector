from src.models import predict_from_url


class FakeNewsDetectorGUI:

    def __init__(self, url_entry, is_box_ticked, canvas, result_title, result_text):
        self.url_entry = url_entry
        self.is_box_ticked = is_box_ticked
        self.canvas = canvas
        self.result_title_id = result_title
        self.result_text_id = result_text

    def classify_url(self):
        """Called when classify button is pressed"""
        url = self.url_entry.get()
        canvas = self.canvas
        print("button")

        prediction = predict_from_url.predict(url, "naive_bayes")

        # detailed = self.tick_var.get()     For once detailed is programmed

        if prediction is None:
            canvas.itemconfig(self.result_title_id, text="URL cannot be read")
            canvas.itemconfig(self.result_text_id, text="Please check the URL you entered was correct and try again")
        else:
            classification = prediction['label']
            certainty = prediction['confidence']
            canvas.itemconfig(self.result_title_id, text=classification)

            # result_text_id.text = 'Put explanation here'
