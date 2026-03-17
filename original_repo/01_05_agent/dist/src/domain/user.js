export function createUser(id, input) {
    return {
        id,
        email: input.email,
        apiKeyHash: input.apiKeyHash,
        createdAt: new Date(),
    };
}
