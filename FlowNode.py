class FlowNode:
    def __init__(self, from_layer_index, from_range_start, from_range_end, to_layer_index, to_range_start, to_range_end):
        self.from_layer_index = from_layer_index
        self.from_range_start = from_range_start
        self.from_range_end = from_range_end

        self.to_layer_index = to_layer_index
        self.to_range_start = to_range_start
        self.to_range_end = to_range_end
