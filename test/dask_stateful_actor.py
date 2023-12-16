from dask.distributed import get_worker, get_client
import os

def receive_message(params):
    sender_url, message, log_file_name = params
    append_to_file(log_file_name, f"Sender_url {sender_url}, Message {message}")

def append_to_file(log_file_name, text):
    if type(text) != str:
        text = str(text)
        
    with open(log_file_name, 'a') as file:
        file.write(text + '\n')

class Actor:
    def __init__(self, params):
        self.worker = get_worker()
        self.client = get_client()

        self.worker_index, self.log_file_path = params

        self.log_file_name = os.path.join(self.log_file_path, "actor_stateful_test_" + str(self.worker_index) + ".txt")

        if os.path.exists(self.log_file_name):
            os.remove(self.log_file_name)

        append_to_file(self.log_file_name, "Actor " + str(self.worker_index) + " initialized.")

    def send_message(self, worker_key, message, log_file_name):
        self.client.submit(receive_message, (self.worker.address, message, log_file_name), workers=worker_key)
        
    def return_worker_index(self, anymessage):
        append_to_file(self.log_file_name, "Worker index: " + str(self.worker_index) + ", Any message: " + anymessage)
        return self.worker_index
    