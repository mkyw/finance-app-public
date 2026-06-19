import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Scope Turbopack to this app dir. The repo has lockfiles at multiple levels;
  // without this, Turbopack guesses a workspace root and warns on each dev start.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
