export function createSession(id, userId, title) {
    return {
        id,
        userId,
        title,
        status: 'active',
        createdAt: new Date(),
    };
}
