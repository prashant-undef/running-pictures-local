from prisma.partials import Face

Face.create_partial("FaceEmbedding", include={"embedding"})
