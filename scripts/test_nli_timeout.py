import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from controlplane.grounding import NliClaimBackend

backend = NliClaimBackend(timeout_s=1.2)

print("warming up (loads/downloads the model, not timed)...")
t0 = time.time()
backend.warm_up()
print(f"warm_up took {time.time()-t0:.2f}s (this is the one-time cost app/api.py should pay at startup)")

claim = "The Eiffel Tower is located in Paris."
context = "The Eiffel Tower is a wrought-iron tower in Paris, France. It was constructed in 1889."

t0 = time.time()
risk, detail = backend.score(claim, context)
print(f"\nscore() after warm_up: risk={risk}  took={time.time()-t0:.2f}s")
print(detail)

claim_response = (
    "The Eiffel Tower is located in Paris. It was built in 1889. "
    "It was designed by Gustave Eiffel. It is made of wrought iron. "
    "It was the tallest structure in the world at the time. "
    "It attracts millions of visitors every year. "
    "It was originally intended as a temporary structure. "
    "It has three levels for visitors."
)
long_context = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was constructed between 1887 and 1889 as the entrance to the 1889 World's Fair. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built it. "
    "At 330 metres tall, it was the tallest man-made structure until 1930. "
    "The tower has three levels open to visitors, with restaurants on the first and second levels. "
    "Around 7 million people visit the tower each year, making it one of the most visited paid monuments in the world."
)

t0 = time.time()
risk, detail = backend.score(claim_response, long_context)
print(f"\nstress test: risk={risk}  took={time.time()-t0:.2f}s")
print(detail)