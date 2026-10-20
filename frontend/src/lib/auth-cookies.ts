/**
 * Auth.js cookie names for this project, prefixed `agri-ml.` instead of the
 * default `authjs.`.
 *
 * Browsers scope cookies by host, NOT by port - so this app (localhost:3001)
 * and the original PRODUCT - ML app (localhost:3000) would otherwise read
 * and overwrite each other's `authjs.session-token`, logging you out of one
 * whenever you sign in to the other. Options mirror Auth.js's own defaults
 * (@auth/core lib/utils/cookie.js defaultCookies).
 */
const useSecureCookies = (process.env.AUTH_URL ?? process.env.NEXTAUTH_URL ?? "").startsWith("https://");
const prefix = useSecureCookies ? "__Secure-" : "";

const base = { httpOnly: true, sameSite: "lax" as const, path: "/", secure: useSecureCookies };
const shortLived = { ...base, maxAge: 60 * 15 };

export const SESSION_COOKIE_NAME = `${prefix}agri-ml.session-token`;
export const SECURE_COOKIES = useSecureCookies;

export const authCookies = {
  sessionToken: { name: SESSION_COOKIE_NAME, options: base },
  callbackUrl: { name: `${prefix}agri-ml.callback-url`, options: base },
  csrfToken: { name: `${useSecureCookies ? "__Host-" : ""}agri-ml.csrf-token`, options: base },
  pkceCodeVerifier: { name: `${prefix}agri-ml.pkce.code_verifier`, options: shortLived },
  state: { name: `${prefix}agri-ml.state`, options: shortLived },
  nonce: { name: `${prefix}agri-ml.nonce`, options: base },
  webauthnChallenge: { name: `${prefix}agri-ml.challenge`, options: shortLived },
};
