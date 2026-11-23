# Writing a PyTorch Conference lightning talk proposal

A lightning talk is 10 minutes and has one idea. Pick the one thing you want people to remember, and cut everything else. The title should name the PyTorch project and the hardware or problem plainly, such as "MXFP8 pretraining with TorchTitan on AMD Instinct GPUs." Skip clever wordplay. Reviewers scan hundreds of titles and pick the ones they understand at once.

Open the abstract with the problem or the open question, not with your company. A strong pattern is "The question is no longer X, but Y," which shows you are past the basics. Then say what you did in one sentence that starts with "In this talk, we present our work..." and names the PyTorch projects you contributed to, such as torchft, Monarch, TorchTitan, or TorchAO. The PyTorch audience cares about the open source stack first and your hardware second, so frame the hardware as the place where that stack now runs.

The middle paragraph should name the concrete pieces you built, using terms the community already knows: a process group, a kernel, a collective, a compiled graph. Avoid internal project names, build tags, and marketing claims. Every piece you name should have public code, an upstream pull request, or a release behind it, because reviewers check, and the AMD and Intel talks all point to upstream work.

End with proof and a takeaway. Give one or two measured numbers from a real run, such as recovery time or throughput, and say what attendees can reuse on their own work. A live demo is a plus, but keep it short enough to fit in 10 minutes. Write the whole abstract as plain prose in two or three short paragraphs, with no bullet lists, in about 150 to 250 words.
