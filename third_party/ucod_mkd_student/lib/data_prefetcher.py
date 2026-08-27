import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target, _, _ ,self.next_target_b,self.next_target_c = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_target_b = None
            self.next_target_c = None
            return

        with torch.cuda.stream(self.stream):
            device = torch.device("cuda:6")
            self.next_input = self.next_input.to(device)
            self.next_target = self.next_target.to(device)
            self.next_target_b = self.next_target_b.to(device)
            self.next_target_c = self.next_target_c.to(device)

            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_target_b = self.next_target_b.float() #if need
            self.next_target_c = self.next_target_c.float() #if need
  


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        target_b=self.next_target_b
        target_c=self.next_target_c
        self.preload()
        return input, target ,target_b,target_c
