from data_manager import DataStream, DataSet, DataSource
from processing import DataProcessor
from data_types import RawData, ProcessedData

def main():
    data_source = None
    try:
        data_source = DataStream("/dev/ttyACM0")
        data_processor = DataProcessor()
        
        print("Collecting data for regression...")
        i=0
        while i<100:
            raw_data = data_source.sample_buffer()
            for data in raw_data:
                if isinstance(data, RawData): # To filter out any None values and any other junk
                    data_processor.add_to_regression(data)
                    i+=1
            print(f"Collected {i} samples")

        data_processor.set_extension_point(data_processor.regression_samples[0])
        data_processor.fit_regression()

        processed_data_list = []
        while True:
            raw_data = data_source.sample()
            if raw_data is None:
                continue
            processed_data = data_processor.process(raw_data)
            processed_data_list.append(processed_data)
            print(f"Processed data: {processed_data}")
            
    finally:
        # Ensure the serial connection is properly closed
        if data_source:
            del data_source

if __name__ == "__main__":
    main()
    
    