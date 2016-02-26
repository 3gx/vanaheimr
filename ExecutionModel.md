# Introduction #

Future architectures are expected to continue to scale exponentially in terms of the number of
computational, interconnect, and memory resources.  However, the performance of individual resources
is expected to scale significantly more slowly.  Applications that can be expected to scale on these
architectures must express a large degree of parallelism: explicit data-parallelism when possible and
hierarchical irregular parallelism when not.  Efficient processors will include a large
number of parallel resources.  It is expected that specialization in terms of the amount of resources devoted to computation, memory, and communication will still be advantageous for tuning a processor
to the requirements of an application.  However, **general purpose processors will likely include a balanced
distribution of resources along with dynamic optimizations that factor out control and data redundancy in
regular applications.**

The Vanaheimr execution model is a specific instantiation of the Multi-BSP model, with enough details
to allow for a concrete implementation.

# Details #

TODO