class Client:
    def __init__(self, customer_id, data_dict):
        self.id_ = customer_id
        for key, value in data_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ', '.join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items() if k != 'id')
        return f"Customer(id={self.id}, {attrs})"