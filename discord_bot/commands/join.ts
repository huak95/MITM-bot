import { SlashCommandBuilder, EmbedBuilder } from "discord.js";
import { joinVoiceChannel, getVoiceConnection } from "@discordjs/voice";
import { v4 as uuidv4 } from "uuid";

// Object to store active sessions
export const activeSessions = new Map();

export const data = new SlashCommandBuilder()
  .setName("join")
  .setDescription("Joins the voice channel and prepares for recording");

export async function execute(interaction) {
  const voiceChannel = interaction.member.voice.channel;
  if (!voiceChannel) {
    return interaction.reply(
      "You need to be in a voice channel to use this command!"
    );
  }

  await interaction.deferReply();

  try {
    let connection = getVoiceConnection(interaction.guild.id);
    if (!connection) {
      connection = joinVoiceChannel({
        channelId: voiceChannel.id,
        guildId: interaction.guild.id,
        adapterCreator: interaction.guild.voiceAdapterCreator,
        selfDeaf: false,
      });
    }

    const sessionId = uuidv4().substring(0, 12);
    const startTime = new Date();

    const sessionInfo = {
      id: sessionId,
      channel: voiceChannel.name,
      startTime: startTime,
      participants: [interaction.user.id],
      activity: [`00:00:00: @${interaction.user.username} joined the session.`],
    };

    activeSessions.set(interaction.guild.id, sessionInfo);

    const embed = new EmbedBuilder()
      .setColor("#0099ff")
      .setTitle("Voice Session Started")
      .addFields(
        { name: "Session ID", value: sessionId },
        { name: "Channel", value: voiceChannel.name },
        { name: "Started", value: startTime.toLocaleTimeString() },
        { name: "Activity", value: sessionInfo.activity[0] }
      )
      .setFooter({ text: "Use /record to start recording" });

    await interaction.editReply({ embeds: [embed] });

    // Set up listener for new participants
    connection.receiver.speaking.on("start", (userId) => {
      if (!sessionInfo.participants.includes(userId)) {
        const user = interaction.guild.members.cache.get(userId);
        const timestamp = new Date();
        const timeDiff = (timestamp - startTime) / 1000;
        const timeDiffFormatted = new Date(timeDiff * 1000)
          .toISOString()
          .substr(11, 8);

        sessionInfo.participants.push(userId);
        sessionInfo.activity.push(
          `${timeDiffFormatted}: @${user.user.username} joined the session.`
        );
      }
    });
  } catch (error) {
    console.error(error);
    await interaction.editReply(
      "There was an error while executing this command!"
    );
  }
}
