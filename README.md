By CraigChrist8239

Keep in mind, I'm new to CUDA. I'm sure there are problems with the class
functions and structure as well. Notice that this does not operate on strings
but uses integer blocks, and allocates bigger blocks on overflow. 

This is going to require a bit of work. Doing almost any operation requires
creating a copy of the object. Before you free an object, you must free its
blocks first. I'm almost positive there are going to be race issues.
I would love feedback on how to improve this.

This is mearly uploaded for any interested parties.
