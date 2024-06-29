import { SlashCommandBuilder } from "discord.js";
import { getVoiceConnection } from "@discordjs/voice";

export const data = new SlashCommandBuilder()
  .setName("stop")
  .setDescription("Stops recording and leaves the voice channel");

export async function execute(interaction) {
  const guildId = interaction.guild.id;
  const connection = getVoiceConnection(guildId);

  if (!connection) {
    return interaction.reply("I'm not currently in a voice channel!");
  }

  try {
    // Destroy the voice connection
    connection.destroy();

    // You might want to do some cleanup here, like stopping any ongoing processes
    // related to recording or saving files.

    await interaction.reply("Stopped recording and left the voice channel.");
  } catch (error) {
    console.error("Error stopping recording:", error);
    await interaction.reply(
      "There was an error while trying to stop the recording."
    );
  }
}
