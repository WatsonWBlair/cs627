"""
Quick script to inspect the format of MOSI words data
"""
import sys
sys.path.insert(0, 'CMU-MultimodalSDK')

from mmsdk import mmdatasdk

data_path = 'data/cmumosi/mosi/'

print("Loading raw words data...")
raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, data_path)

words_seq = raw_data.computational_sequences.get('words')

if words_seq:
    # Get first segment ID
    first_seg = list(words_seq.data.keys())[0]
    print(f"\nFirst segment ID: {first_seg}")

    # Inspect the data structure
    seg_data = words_seq.data[first_seg]
    print(f"\nKeys in segment data: {seg_data.keys()}")

    words_features = seg_data['features']
    print(f"\nType of features: {type(words_features)}")
    print(f"Shape: {words_features.shape if hasattr(words_features, 'shape') else 'N/A'}")
    print(f"\nFirst few elements:")
    print(words_features[:5] if hasattr(words_features, '__getitem__') else words_features)

    # Try to decode if it's bytes
    if hasattr(words_features[0], 'decode'):
        print(f"\nDecoded words: {[w.decode('utf-8') for w in words_features[:10]]}")

    # Check intervals too
    if 'intervals' in seg_data:
        intervals = seg_data['intervals']
        print(f"\nIntervals shape: {intervals.shape if hasattr(intervals, 'shape') else len(intervals)}")
        print(f"First interval: {intervals[0]}")
else:
    print("Words sequence not found!")
    print(f"Available sequences: {list(raw_data.computational_sequences.keys())}")
