Vanheimr is an aggressive project targeting a high efficiency high performance computing platform.  The idea is to create an open hardware architecture, compiler infrastructure, and program representation that enables radical departures from commercial approaches.

# News #

## On Holes ##

All too often new designs begin by throwing out the lessons learned
from generations of innovation in the hopes of ultimately leading to
a more tenable solution.  In the context of system design, new projects
may be quick to discard existing RTL, architecture, compiler, operating system,
application, or algorithmic methodologies at the first sign of trouble.  Unfortunately
the new designs that arise as replacements are as likely to be cast aside by the
filter of time as the multitude of potential solutions that were conceived, evaluated, discarded,
and ultimately faded from memory during the development of the previous system.

This doesn't mean that changing assumptions cannot demand changes in implementation.
New discoveries and improved understanding of the physics of electrical devices
and algorithms that can be applied to important computing problems are likely to
expand or constrain the space of tenable and efficient systems, possibly blocking off
avenues of development that once seemed promising and mandating expansion and innovation
in new directions.

In these cases, it is important to reevaluate the problem of system design with a firm understanding of the changed
assumptions, knowledge of the best solutions under the previous set of assumptions, and a large
corpus of attempted, but failed, prior work in mind.  At all times, it is necessary to evaluate a new design
in the context the best available solutions using existing systems.  Much of the time, important results
from prior work (positive and negative) will not be invalidated by new assumptions.  We typically refine
our understanding, or rather, we hardly ever turn the world upside down.

However, still, in many cases hard problems will persist that defy existing solutions and related work.
These are holes in the completeness of our new system.   Existing systems working under different
assumptions did not have these short comings, but our new solution does.  Hopefully, if we press on in these cases, the filter of time will tell whether new assumptions necessarily lead to
inferior solutions, or whether a good solution requires forging ahead into new territory.
Too many holes will sink a system, but we often spend our time reinventing old solutions
to trival problems, or chasing after up-to-now untenable solutions to hard problems.

In the spirit of focusing attention away from re-implementing well known aspects of a system stack for Vanaheimr, [this page](http://code.google.com/p/vanaheimr/wiki/Holes) attempts to characterize holes in the design that have elegant and efficient solutions in the context of contemporary systems.  Mostly it focuses
on holes that were difficult to fill by previous systems, with a long history of minor results building up  over decades to the current solution.

# Core Features #
  * Hierarchical SIMD processor organization
  * Software stack written in CUDA
    * Compiler
    * Architecture Simulator
    * Runtime/Operating System

It is expected that the development of these components will require innovations in algorithm design and novel
software engineering practices.

# Authors #

  * Gregory Diamos
  * Sudnya Diamos