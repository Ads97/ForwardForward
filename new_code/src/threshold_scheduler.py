class Threshold_Scheduler:
    
    def __init__(self, initial_multiplier, rate=1.1, patience=1):
        self.multiplier = initial_multiplier
        self.rate = rate
        self.patience = patience
        
        
    def step(self, epoch):
        if epoch != 0 and epoch % self.patience == 0:
            self.multiplier *= self.rate
        print("Setting Threshold: ", self.multiplier)
    
        