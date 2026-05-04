import gc
from modules.data_loader import DataLoader
from modules.data_retriever import DataRetriever


if __name__ == "__main__":
    data_loader = DataLoader(files_dir="data", images_dir="figures")
    data_retriever = DataRetriever(data_loader=data_loader, persist_dir="storage")
    del data_retriever; gc.collect()