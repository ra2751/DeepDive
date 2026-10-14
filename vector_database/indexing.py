class IndexManager:

    @staticmethod
    def create_flat_index(collection):

        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

    @staticmethod
    def create_hnsw_index(collection):

        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

    @staticmethod
    def create_ivf_index(collection):

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 128
            }
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )